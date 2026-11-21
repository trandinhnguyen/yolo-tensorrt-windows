# Security policy

## Supported versions

Security fixes are made on the `main` branch and included in the next release.

## Reporting a vulnerability

Please **don't** report security issues in public issues or pull requests.

Use GitHub's private vulnerability reporting instead: open the
[Security tab](https://github.com/trandinhnguyen/yolo-tensorrt-windows/security)
of this repository and click **Report a vulnerability**. Include the steps to
reproduce, the affected version or commit, and the impact you expect.

You should get a first response within a week. Once the issue is confirmed and
fixed, the advisory will be published with credit to you, unless you'd rather
stay anonymous.

## Scope

This app processes local video files on your machine and doesn't open network
ports. Relevant reports include, for example, crafted video or model files that
lead to code execution, and unsafe handling of downloaded dependencies in
`scripts/setup.ps1`. Vulnerabilities in TensorRT, CUDA, FFmpeg, Qt or other
third-party components should go to their respective maintainers.
