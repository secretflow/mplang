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
import uuid
from typing import Any

import cloudpickle as pickle
import httpx

from mplang.core.comm import CommunicatorBase

logger = logging.getLogger(__name__)


class HttpCommunicator(CommunicatorBase):
    def __init__(self, session_name: str, rank: int, endpoints: list[str]):
        super().__init__(rank, len(endpoints))
        self.session_name = session_name
        self.endpoints = endpoints
        logger.info(
            f"HttpCommunicator initialized: session={session_name}, rank={rank}, endpoints={endpoints}"
        )

    def new_id(self) -> str:
        return str(uuid.uuid4())

    def send(self, to: int, key: str, data: Any) -> None:
        """Sends data to a peer party by POSTing to its /comm/send endpoint."""
        target_endpoint = self.endpoints[to]
        url = f"{target_endpoint}/sessions/{self.session_name}/comm/send"
        logger.info(
            f"Sending data: from_rank={self._rank}, to_rank={to}, key={key}, target_url={url}"
        )
        logger.debug(f"Data to send: {data}")

        try:
            # Use cloudpickle for robust serialization of complex Python objects
            data_bytes = pickle.dumps(data)
            data_b64 = base64.b64encode(data_bytes).decode("utf-8")

            request_data = {
                "from_rank": self._rank,
                "to_rank": to,
                "data": data_b64,
                "key": key,
            }
            logger.debug(f"Request payload: {request_data}")

            response = httpx.post(url, json=request_data, timeout=60)
            logger.info(f"Send response: status={response.status_code}")
            if response.status_code != 200:
                logger.error(f"Send failed: {response.text}")
            response.raise_for_status()
            logger.info(
                f"Send completed successfully: from_rank={self._rank}, to_rank={to}, key={key}"
            )
        except httpx.RequestError as e:
            logger.error(
                f"Send failed with exception: from_rank={self._rank}, to_rank={to}, key={key}, error={e}"
            )
            raise OSError(f"Failed to send data to rank {to}") from e

    def recv(self, frm: int, key: str) -> Any:
        """Wait until the key is set, returns the value. Override to add logging."""
        logger.info(
            f"Waiting to receive: from_rank={frm}, to_rank={self._rank}, key={key}"
        )
        # The actual data is stored as bytes, so we need to deserialize it
        data_bytes = super().recv(frm, key)
        result = pickle.loads(data_bytes)
        logger.info(f"Received data: from_rank={frm}, to_rank={self._rank}, key={key}")
        logger.debug(f"Received data content: {result}")
        return result

    # recv() method is inherited from CommunicatorBase
