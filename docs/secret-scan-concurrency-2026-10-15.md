# Superseded secret-scan queue repair

Secret scan checks the complete tracked tree on every push, pull request and
manual dispatch. Previously, checkpoint pushes left obsolete scans queued even
when a newer scan already covered the resulting tree.

The workflow now groups runs by workflow name and PR number or ref, with
`cancel-in-progress: true`. A newer run replaces queued or running checks of
the same context. Different branches, tags and PRs remain independent.
All triggers, read-only permissions, the five-minute timeout, credential
patterns, account-name handling and the four existing tests are unchanged.

Local checks pass: all four secret guards unskipped, and the existing workflow
linter parses all 437 bash steps. A parsed-YAML comparison confirms that only
the concurrency metadata changes. Two committed branch checkpoints exercise
actual hosted replacement; the terminal metadata result will be recorded
before integration. A queued latest scan is not a hosted test pass.
