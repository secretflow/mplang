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

import logging

import grpc
from google.protobuf.json_format import MessageToJson, Parse

try:
    import trustflow.attestation.generation as tdx_generation
except ImportError:
    tdx_generation = None

try:
    from secretflowapis.v2.sdc.ual_pb2 import (
        UnifiedAttestationGenerationParams,
        UnifiedAttestationReport,
    )
except ImportError:
    UnifiedAttestationGenerationParams = None
    UnifiedAttestationReport = None
    IntelTdxReport = None

from mplang.protos import executor_pb2, executor_pb2_grpc


class TEEAttestationService(executor_pb2_grpc.TEEAttestationServiceServicer):
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

    def GetTEEReport(
        self, request: executor_pb2.GetTEEReportRequest, context: grpc.ServicerContext
    ) -> executor_pb2.GetTEEReportResponse:
        """Generate TEE attestation report."""
        logging.info(f"GetTEEReport: {request.name}")

        try:
            report_json, public_key = self._generate_report(request)

            response = executor_pb2.GetTEEReportResponse()
            response.pem_public_key = public_key
            response.tee_mode = self._tee_mode
            response.report_json = report_json

            return response

        except Exception as e:
            logging.exception(f"Failed to generate TEE report: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to generate TEE report: {e}")
            return executor_pb2.GetTEEReportResponse()

    def _generate_report(
        self, request: executor_pb2.GetTEEReportRequest
    ) -> tuple[str, str]:
        """Generate TEE attestation report."""

        public_key = self._key_manager.get_or_create_key_pair(request.name)

        params = UnifiedAttestationGenerationParams()
        Parse(request.generation_params_json, params)

        if self._tee_mode == "sim":
            report_json = self._generate_sim_report()
        elif self._tee_mode == "tdx":
            report_json = self._generate_tdx_report(params)
        else:
            raise ValueError(f"Unsupported TEE mode: {self._tee_mode}")

        return report_json, public_key

    def _generate_sim_report(self) -> str:
        # Create simulation report
        return "dummy"

    def _generate_tdx_report(self, params: UnifiedAttestationGenerationParams) -> str:
        # Generate TDX report
        generator = tdx_generation.create_attestation_generator()
        report_json: str = generator.generate_report_json(MessageToJson(params))

        return report_json


class TEEKeyManager:
    """Manager for TEE key pairs."""

    def __init__(self):
        self._keys: dict[str, str] = {}  # tee_identity -> public_key_pem

    def get_or_create_key_pair(self, session_name: str) -> str:
        """Generate RSA key pair for TEE identity."""
        if session_name in self._keys:
            return self._keys[session_name]

        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import rsa

            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )

            # Serialize public key
            public_key = private_key.public_key()
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("utf-8")

            self._keys[session_name] = public_pem
            return public_pem

        except ImportError:
            # Fallback for environments without cryptography
            return (
                "-----BEGIN PUBLIC KEY-----\nSIMULATED_KEY\n-----END PUBLIC KEY-----\n"
            )


def add_tee_attestation_service(
    server, key_manager: TEEKeyManager, tee_mode: str = "sim"
) -> None:
    """Add TEE attestation service to gRPC server."""
    service = TEEAttestationService(key_manager, tee_mode)
    executor_pb2_grpc.add_TEEAttestationServiceServicer_to_server(service, server)
    logging.info(f"TEE attestation service added (mode: {tee_mode})")
