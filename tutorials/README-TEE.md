# TEE Attestation Tutorial (Mock)

This tutorial demonstrates a mock TEE attestation flow with payload-based quotes
and a mock crypto stream cipher. It's for plumbing and API demonstration only.

Notes:

- Use UINT8 arrays for plaintext in this demo; the mock crypto backend only
  supports byte arrays and prepends a 12-byte nonce to ciphertext.
- The TEE backend uses a trivial quote format and does not implement real
  vendor attestation yet.

Run tee_attestation.py with the in-memory Simulator to see the end-to-end flow.
