# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TEE attestation service implementation.

This module provides the TEE attestation functionality for generating
attestation reports in both simulation and TDX modes.
"""

from __future__ import annotations

import base64
import functools
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import grpc
import trustflow.attestation.verification as verification
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from google.protobuf.json_format import MessageToJson

try:
    import trustflow.attestation.generation as tdx_generation
except ImportError:
    tdx_generation = None

try:
    from secretflowapis.v2.sdc.ual_pb2 import (
        UnifiedAttestationAttributes,
        UnifiedAttestationGenerationParams,
        UnifiedAttestationPolicy,
        UnifiedAttestationReport,
        UnifiedAttestationReportParams,
    )
except ImportError:
    UnifiedAttestationGenerationParams = None
    UnifiedAttestationReport = None
    UnifiedAttestationReportParams = None
    UnifiedAttestationPolicy = None
    UnifiedAttestationAttributes = None

from mplang.protos.v1alpha1 import tee_pb2, tee_pb2_grpc


def grpc_exception_handler(
    response_cls: type[Any],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(self: Any, request: Any, context: grpc.ServicerContext) -> Any:
            try:
                return func(self, request, context)
            except ValueError as be:
                context.set_details(str(be))
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return response_cls()
            except grpc.RpcError as e:
                context.set_code(e.code())
                context.set_details(f"Failed to call rpc: {e.details()}")
                return response_cls()
            except Exception:
                context.set_details("Internal error.")
                context.set_code(grpc.StatusCode.INTERNAL)
                return response_cls()

        return wrapper

    return decorator


class TEEAttestationService(tee_pb2_grpc.TEEMgrServiceServicer):
    """gRPC service implementation for TEE attestation."""

    def __init__(self, key_manager: TEEKeyManager, tee_mode: str = "sim"):
        """
        Initialize TEE attestation service.

        Args:
            tee_mode: TEE mode, either "sim" or "tdx"
        """
        self._tee_mode = tee_mode
        self._key_manager = key_manager

        if self._tee_mode == "tdx" and tdx_generation is None:
            raise ImportError(
                "trustflow-tdx-generation package is required for TDX mode"
            )

        if self._tee_mode == "tdx" and UnifiedAttestationReport is None:
            raise ImportError("sdc-apis package is required for TEE attestation")

    @grpc_exception_handler(tee_pb2.InitResponse)
    def Init(
        self, request: tee_pb2.InitRequest, context: grpc.ServicerContext
    ) -> tee_pb2.InitResponse:
        """Initialize TEE session and verify attestation."""
        logging.info(f"Init TEE session: {request.session_name}")

        response = tee_pb2.InitResponse()

        # Create a gRPC channel to the TEE party
        channel = grpc.insecure_channel(request.tee_party_addr)
        stub = tee_pb2_grpc.TEEMgrServiceStub(channel)

        report_request = tee_pb2.GetTEEReportRequest()
        report_request.session_name = request.session_name
        report_request.pem_public_key, _ = (
            self._key_manager.get_or_create_self_key_pair(request.session_name)
        )
        report_request.nonce = base64.b64encode(os.urandom(16)).decode("utf-8")

        # Call the TEE party's GetTEEReport method
        report_response = stub.GetTEEReport(report_request)
        if report_response.tee_mode != self._tee_mode:
            raise ValueError(
                f"TEE mode mismatch: expected {self._tee_mode}, got {report_response.tee_mode}"
            )
        # store tee party pub key
        if not report_response.pem_public_key:
            raise ValueError(
                "TEE party did not return a public key in the attestation report."
            )
        self._key_manager.insert_tee_pub_key(
            request.session_name, report_response.pem_public_key
        )

        if report_response.tee_mode == "sim":
            logging.warning("Running in sim mode, skipping attestation verification")
            return response

        attrs = UnifiedAttestationAttributes()
        attrs.str_tee_platform = "TDX"
        attrs.hex_user_data = report_request.nonce
        # TODO: use user provide measurement
        # attrs.hex_ta_measurement = "test"
        attrs.bool_debug_disabled = "true"
        status = verification.attestation_report_verify(
            report_response.report_json,
            UnifiedAttestationPolicy(main_attributes=[attrs]),
        )
        if status.code != 0:
            raise ValueError(
                f"Attestation verification failed: {status.message}, detail: {status.detail}"
            )

        return response

    @grpc_exception_handler(tee_pb2.GetTEEReportResponse)
    def GetTEEReport(
        self, request: tee_pb2.GetTEEReportRequest, context: grpc.ServicerContext
    ) -> tee_pb2.GetTEEReportResponse:
        """Generate TEE attestation report."""
        logging.info(f"GetTEEReport: {request.session_name}")

        public_key, _ = self._key_manager.get_or_create_self_key_pair(
            request.session_name
        )

        report_json = self._generate_report(request)

        response = tee_pb2.GetTEEReportResponse()
        response.pem_public_key = public_key
        response.tee_mode = self._tee_mode
        response.report_json = report_json

        # store non-tee party pub key
        self._key_manager.insert_peer_pub_key(
            request.session_name, request.rank, request.pem_public_key
        )

        return response

    def _generate_report(self, request: tee_pb2.GetTEEReportRequest) -> str:
        """Generate TEE attestation report."""

        params = UnifiedAttestationGenerationParams()
        params.tee_identity = "tdx_instance"
        params.report_type = "Passport"
        params.report_params = UnifiedAttestationReportParams()
        params.report_params.hex_user_data = request.nonce

        if self._tee_mode == "sim":
            report_json = self._generate_sim_report()
        elif self._tee_mode == "tdx":
            report_json = self._generate_tdx_report(params)
        else:
            raise ValueError(f"Unsupported TEE mode: {self._tee_mode}")

        return report_json

    def _generate_sim_report(self) -> str:
        # Create simulation report
        return "dummy"

    def _generate_tdx_report(self, params: Any) -> str:
        # Generate TDX report
        generator = tdx_generation.create_attestation_generator()
        report_json: str = generator.generate_report_json(MessageToJson(params))

        return report_json


@dataclass
class SessionKeySet:
    pub_key: str
    priv_key: str
    peers_pub_key: dict[int, str]


class TEEKeyManager:
    """Manager for TEE key pairs."""

    def __init__(self) -> None:
        self._keys: dict[str, SessionKeySet] = {}
        self._tee_keys: dict[str, str] = {}

    def insert_peer_pub_key(self, session_name: str, rank: int, pub_key: str) -> None:
        if session_name not in self._keys:
            raise ValueError(f"Can not found key pair for session: {session_name}")
        self._keys[session_name].peers_pub_key[rank] = pub_key

    def get_or_create_self_key_pair(self, session_name: str) -> tuple[str, str]:
        if session_name in self._keys:
            return self._keys[session_name].pub_key, self._keys[session_name].priv_key

        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        # Serialize public key
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

        self._keys[session_name] = SessionKeySet(
            pub_key=public_pem, priv_key=private_pem, peers_pub_key={}
        )
        return public_pem, private_pem

    def insert_tee_pub_key(self, session_name: str, pub_key: str) -> None:
        self._tee_keys[session_name] = pub_key

    def get_tee_pub_key(self, session_name: str) -> str:
        if session_name not in self._tee_keys:
            raise ValueError(f"Can not found tee pub key for session:{session_name}")
        return self._tee_keys[session_name]

    def get_session_key_dict(self, session_name: str) -> dict[int, str]:
        if session_name not in self._keys:
            raise ValueError(f"Can not found session:{session_name}")
        return self._keys[session_name].peers_pub_key

    def get_peer_pub_key(self, session_name: str, rank: int) -> str:
        if session_name not in self._keys:
            raise ValueError(f"Can not found session:{session_name}")
        if rank not in self._keys[session_name].peers_pub_key:
            raise ValueError(
                f"Can not found rank:{rank} pub key in session:{session_name}"
            )
        return self._keys[session_name].peers_pub_key[rank]


def add_tee_attestation_service(
    server: Any, key_manager: TEEKeyManager, tee_mode: str = "sim"
) -> None:
    """Add TEE attestation service to gRPC server."""
    service = TEEAttestationService(key_manager, tee_mode)
    tee_pb2_grpc.add_TEEMgrServiceServicer_to_server(service, server)
    logging.info(f"TEE attestation service added (mode: {tee_mode})")
