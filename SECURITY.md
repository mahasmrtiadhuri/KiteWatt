# Security policy and boundaries

This is an experimental local app. It is **not a hardened public API**.

- Keep the dashboard bound to `127.0.0.1`. Do not expose its port on the internet or an untrusted LAN.
- Do not load skill files from unknown sources; they contain data rather than executable source, but can be forged and are not cryptographically signed.
- The tiny DSL supports only named allowlisted operators and bounded inputs; no arbitrary code execution is supported.
- Network requests to paid model providers are not part of the project.
- Do not treat a finite set of validation examples as a formal proof of a skill's correctness.

For reporting concerns, open a GitHub issue without including secrets, private datasets, or personal information.
