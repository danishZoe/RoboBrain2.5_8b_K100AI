---
name: github-repo-push
description: Push a repository that lives on a remote SSH server to GitHub using the local machine's existing SSH or GitHub credentials. Use when the user explicitly asks to publish or synchronize a remote repository.
metadata:
  short-description: Push remote work to GitHub safely
---

# GitHub repository push

Use this workflow when the working repository is on a remote server but GitHub authentication is already configured on the local machine.

## Preconditions

- Confirm the user explicitly authorized the push and identify the exact GitHub repository and branch.
- Inspect the remote repository status and latest commit before exporting it.
- Check local GitHub authentication without printing private keys or tokens. Prefer an existing SSH host alias such as `github.com-danishZoe`.
- Do not create, copy, or persist a GitHub token on the remote server when local SSH authentication is available.

## Preferred transfer workflow

1. On the remote server, verify the repository is clean enough to publish and record the commit:

   ```bash
   cd /root/private_data/workspace/robo
   git status --short
   git log -1 --oneline
   git bundle create tmp/repo-push.bundle --all
   ```

2. Copy the bundle to the local machine with `scp`. Keep it under a task-local temporary directory.

3. Clone the bundle locally. A clone made from a bundle may set `origin` to the bundle path; replace it before pushing:

   ```bash
   git clone repo-push.bundle push-repo
   git -C push-repo remote set-url origin git@github.com-danishZoe:OWNER/REPO.git
   git -C push-repo branch -M master
   ```

4. Push the requested branch using the local SSH configuration:

   ```bash
   git -C push-repo push -u origin master
   ```

5. Verify the remote ref resolves to the same commit:

   ```bash
   git ls-remote git@github.com-danishZoe:OWNER/REPO.git refs/heads/master
   git -C push-repo rev-parse HEAD
   ```

## Authentication checks

Use a non-interactive SSH probe before pushing:

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -T git@github.com-danishZoe
```

Successful GitHub authentication normally prints a greeting and exits with a nonzero status because GitHub does not provide shell access. Treat the greeting as authentication evidence; do not interpret the exit code alone as failure.

If local SSH authentication is unavailable, stop and ask the user for either a repository-scoped fine-grained PAT or an SSH key setup. Never request a personal private key and never put a token in a remote command, URL, repository file, or shell history.

## Scope and safety

- Push only the branch and repository named by the user.
- Do not force-push, rewrite history, delete remote branches, or make the repository public unless separately authorized.
- Exclude model weights, caches, virtual environments, temporary outputs, secrets, and large benchmark mirrors with `.gitignore` before publishing.
- After publishing, report the repository URL, branch, commit hash, authentication method, and verification result.
