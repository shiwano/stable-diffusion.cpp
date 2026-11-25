---
name: sync-upstream
description: Update a local stable-diffusion.cpp branch to the latest upstream/master and verify which local patches were preserved, changed, or incorporated upstream. Also use when reviewing patches after a sync.
---

## Sync the branch

1. Check the current branch, working tree, submodules, and any Git operation in progress. Preserve uncommitted work. For a verification-only request, identify the pre-rebase and post-rebase commits in the reflog and go directly to verification.
2. Record the current HEAD as `old_head`. Confirm the upstream URL and run `git fetch upstream master`. Record the SHA of FETCH_HEAD as `new_base` and the merge base of `old_head` and `new_base` as `old_base`. If upstream history was rewritten, verify the original fork point.
3. Inspect the original patches with `git cherry -v <new_base> <old_head> <old_base>`. A minus sign indicates an equivalent patch upstream. A patch marked with a plus sign may still have landed in a squash or through a different implementation.
4. If HEAD already contains `new_base`, the branch is up to date. If a fast-forward is possible, run `git merge --ff-only <new_base>`. If the branches have diverged, replay the local patches with `git rebase --onto <new_base> <old_base>`. Preserve the topology of any local merge commits. When resolving conflicts, inspect the original patch and the upstream changes and preserve the intent of both. Ask the user if the resolution requires a decision about intended behavior.
5. For each changed gitlink, check the submodule for local changes before updating it to the recorded SHA.

## Verify the patches

- Compare the two patch series with `git range-diff <old_base>..<old_head> <new_base>..<new_head>`. For each entry marked `!`, inspect the diff for behavior affected by conflict resolution. For patches appearing on only one side, compare the original changes with the resulting code. When reporting that upstream absorbed a patch, cite the upstream commit or implementation.
- Confirm that HEAD contains `new_base`, check the working tree and submodule status, and run `git diff --check <new_base>..<new_head>`. Choose a focused build or test for behavior touched by conflict resolution.
- Report the upstream revision, the new HEAD, which local patches were preserved, changed, or incorporated upstream, how conflicts were resolved, verification results, and any tests left unrun.
