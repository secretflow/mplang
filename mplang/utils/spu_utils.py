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

"""SPU-related utility functions for protocol and field type conversion."""

import spu.libspu as libspu

# Global mappings for SPU protocol and field type conversion
SPU_PROTOCOL_MAPPING = {
    "REF2K": libspu.ProtocolKind.REF2K,
    "SEMI2K": libspu.ProtocolKind.SEMI2K,
    "ABY3": libspu.ProtocolKind.ABY3,
    "CHEETAH": libspu.ProtocolKind.CHEETAH,
    "SECURENN": libspu.ProtocolKind.SECURENN,
}

SPU_FIELD_MAPPING = {
    "FM32": libspu.FieldType.FM32,
    "FM64": libspu.FieldType.FM64,
    "FM128": libspu.FieldType.FM128,
}


def parse_protocol(protocol: str | int) -> libspu.ProtocolKind:
    """Parse SPU protocol from string or integer to ProtocolKind enum.

    Args:
        protocol: Protocol specification as string (e.g., "SEMI2K") or integer.

    Returns:
        libspu.ProtocolKind: The corresponding protocol enum.

    Raises:
        ValueError: If the protocol is invalid.

    Examples:
        >>> parse_spu_protocol("SEMI2K")
        ProtocolKind.SEMI2K
        >>> parse_spu_protocol(2)
        ProtocolKind.SEMI2K
    """
    if isinstance(protocol, str):
        if protocol not in SPU_PROTOCOL_MAPPING:
            raise ValueError(
                f"Invalid SPU protocol: {protocol}. "
                f"Valid protocols are: {list(SPU_PROTOCOL_MAPPING.keys())}"
            )
        return SPU_PROTOCOL_MAPPING[protocol]
    else:
        # Assume it's an integer, validate it
        try:
            spu_protocol = libspu.ProtocolKind(protocol)
            # Check if it's a valid enum value (not ???)
            if spu_protocol.name == "???":
                raise ValueError(f"Invalid SPU protocol value: {protocol}")
            return spu_protocol
        except TypeError as exc:
            raise ValueError(
                f"Invalid SPU protocol: {protocol}. "
                f"Must be a valid protocol string or integer."
            ) from exc


def parse_field(field: str | int) -> libspu.FieldType:
    """Parse SPU field type from string or integer to FieldType enum.

    Args:
        field: Field type specification as string (e.g., "FM64") or integer.

    Returns:
        libspu.FieldType: The corresponding field type enum.

    Raises:
        ValueError: If the field type is invalid.

    Examples:
        >>> parse_spu_field("FM64")
        FieldType.FM64
        >>> parse_spu_field(2)
        FieldType.FM64
    """
    if isinstance(field, str):
        if field not in SPU_FIELD_MAPPING:
            raise ValueError(
                f"Invalid SPU field type: {field}. "
                f"Valid field types are: {list(SPU_FIELD_MAPPING.keys())}"
            )
        return SPU_FIELD_MAPPING[field]
    else:
        # Assume it's an integer, validate it
        try:
            spu_field = libspu.FieldType(field)
            # Check if it's a valid enum value
            if spu_field.name == "???":
                raise ValueError(f"Invalid SPU field type value: {field}")
            return spu_field
        except TypeError as exc:
            raise ValueError(
                f"Invalid SPU field type: {field}. "
                f"Must be a valid field type string or integer."
            ) from exc


def list_protocols() -> list[str]:
    """Get list of valid SPU protocol names.

    Returns:
        List of valid protocol names as strings.
    """
    return list(SPU_PROTOCOL_MAPPING.keys())


def list_fields() -> list[str]:
    """Get list of valid SPU field type names.

    Returns:
        List of valid field type names as strings.
    """
    return list(SPU_FIELD_MAPPING.keys())
