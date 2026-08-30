# GeoLLaVA integration status

The audited experiment host was based on GeoLLaVA-8K commit
`a52786cfa3efe984a64ecf3188821cb0b027a0d6`.

The host-integration patch is intentionally excluded from the public Git tree.
The upstream repository did not expose a license during the release audit, so its
redistribution and derivative-patch terms require written clarification from
the upstream rights holder. Any provenance copy must remain outside this public
release.

Once permission is documented, restore a minimal patch for:

- `longva/longva/model/llava_arch.py`
- `longva/longva/model/multimodal_encoder/clip_encoder.py`

Then verify it with `git apply --check` against the exact commit above and add a
one-sample host integration test. The authorized integration must expose
`_dualcomp_host_integration_version = "paper-v1"` on the loaded host model so
the public evaluator can fail closed when the compression hook is absent. Until
then, this repository is a release candidate for the DualComp component, not an
end-to-end paper reproduction.
