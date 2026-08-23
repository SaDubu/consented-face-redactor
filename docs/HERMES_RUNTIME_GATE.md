# Hermes Runtime Gate

```text
Status: UNRESOLVED
Implementation permission: DENIED
Project source task: NOT READY
```

`Hermes` is currently a name supplied by the Human Owner, not a fixed executable, model, repository, or security boundary. No agent receives filesystem, shell, Git, network, credential, or model permission from this document alone.

## 1. Required identity record

Complete and review these fields before installation or invocation:

```yaml
agent_distribution: TBD
source_repository: TBD
release_or_commit: TBD
model_provider: TBD
model_tag: TBD
runner: TBD
runner_version: TBD
endpoint: TBD_LOCAL_LOOPBACK_ONLY
context_length: TBD
worker_git_name: TBD
worker_git_email: TBD_NON_PERSONAL
```

## 2. Probe sequence

1. Verify package/release provenance and license without running the agent.
2. Verify the model endpoint binds to loopback only and does not require a credential in prompts or repository files.
3. Run a read-only probe against a disposable non-project file with all other tools denied.
4. Run a one-file write canary in an isolated worktree with network and arbitrary shell denied.
5. Permit only deterministic checks, a worker-authored commit, and a branch-specific non-force push after Human approval.
6. Independently verify base ancestry, changed paths, commit author, remote SHA, command results, and secret/private-data scans.

## 3. Mandatory deny policy

- no external-directory access
- no arbitrary shell or PowerShell
- no package installation during a task
- no web or Discord-triggered execution
- no credential reads
- no `main` mutation, merge, rebase, reset, clean, tag, or force push
- no model reasoning or raw prompt stored as evidence
- no automatic next-stage execution

## 4. Canary pass criteria

The canary passes only when the agent itself invokes native tools, changes exactly the declared file, executes only allowed commands, creates the expected worker commit, and reaches only the assigned remote branch. A textual tool request, supervisor-applied patch, unexpected permission prompt, extra path, failed check, or missing push is `BLOCKED`.

A passing canary permits only a separately approved low-risk documentation or scaffold task. It does not authorize model download, biometric data access, dependency changes, implementation beyond the exact Task Contract, merge, release, or deployment.
