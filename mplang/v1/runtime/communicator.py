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
This module provides a client-side communicator for interacting with the HTTP backend.
Its sole responsibility is to handle inter-party data exchange (send/recv).
"""

import base64
import logging
from typing import Any

import httpx

from mplang.v1.core.comm import CommunicatorBase
from mplang.v1.kernels.value import Value, decode_value, encode_value


class HttpCommunicator(CommunicatorBase):
    def __init__(self, session_name: str, rank: int, endpoints: list[str]):
        # Validate endpoints
        if not endpoints:
            raise ValueError("endpoints cannot be empty")

        if not all(endpoint for endpoint in endpoints):
            raise ValueError("endpoints cannot contain empty elements")

        super().__init__(rank, len(endpoints))
        self.session_name = session_name
        # Ensure all endpoints have protocol prefix
        self.endpoints = [
            endpoint
            if endpoint.startswith(("http://", "https://"))
            else f"http://{endpoint}"
            for endpoint in endpoints
        ]
        self._counter = 0
        logging.info(
            f"HttpCommunicator initialized: session={session_name}, rank={rank}, endpoints={self.endpoints}"
        )

    # override
    def new_id(self) -> str:
        res = self._counter
        self._counter += 1
        return str(res)

    def send(self, to: int, key: str, data: Any) -> None:
        """Sends data to a peer party by PUTing to its /comm/{key}/from/{from_rank} endpoint."""
        target_endpoint = self.endpoints[to]
        url = f"{target_endpoint}/sessions/{self.session_name}/comm/{key}/from/{self._rank}"
        logging.debug(
            f"Sending data: from_rank={self._rank}, to_rank={to}, key={key}, target_url={url}"
        )

        try:
            # Serialize data using Value envelope.
            if not isinstance(data, Value):
                raise TypeError(
                    f"Communicator requires Value instance, got {type(data).__name__}. "
                    "Wrap data in TensorValue or custom Value subclass."
                )
            data_bytes = encode_value(data)

            data_b64 = base64.b64encode(data_bytes).decode("utf-8")

            request_data = {
                "data": data_b64,
            }

            response = httpx.put(url, json=request_data, timeout=60)
            logging.debug(f"Send response: status={response.status_code}")
            if response.status_code != 200:
                logging.error(f"Send failed: {response.text}")
            response.raise_for_status()
        except httpx.RequestError as e:
            logging.error(
                f"Send failed with exception: from_rank={self._rank}, to_rank={to}, key={key}, error={e}"
            )
            raise OSError(f"Failed to send data to rank {to}") from e

    def recv(self, frm: int, key: str) -> Any:
        """Wait until the key is set, returns the value. Override to add logging."""
        logging.debug(
            f"Waiting to receive: from_rank={frm}, to_rank={self._rank}, key={key}"
        )
        data_b64 = super().recv(frm, key)

        data_bytes = base64.b64decode(data_b64)
        # Deserialize using Value envelope
        result = decode_value(data_bytes)

        logging.debug(
            f"Received data: from_rank={frm}, to_rank={self._rank}, key={key}"
        )
        return result
