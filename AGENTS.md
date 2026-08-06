@AGENTS.md

# Bash commands
- `uv run pytest`: Run the tests.
- `make style && make quality` run the linter + formatter.
- `uv run mkdocs serve`: View the documentation locally at http://127.0.0.1:8000/
- `uv run mkdocs build`: Build the documentation to the `site/` directory.



# Workflow
- When creating a PR, always add a summary to `CHANGELOG.md` with a link to the PR (e.g., `- Description of change (https://github.com/allenai/open-instruct/pull/123).`).
- Always run the linter and make sure the tests pass before finishing a task.
- Prefer running single tests, not the whole suite, when developing.
- To run the `./scripts/train/build_image_and_launch.sh` script, you must commit the current changes.
- To launch experiment scripts, use the `build_image_and_launch.sh` script, like this: `./scripts/train/build_image_and_launch.sh $SOME_SCRIPT`.
- For GRPO, we have three test scripts:
  - `scripts/train/debug/single_gpu_on_beaker.sh`: single GPU, no tools (~8 minutes).
  - `scripts/train/debug/tools/olmo_3_parser_multigpu.sh`: multi GPU, with tools.
  - `scripts/train/debug/large_test_script.sh`: two 8x GPU nodes, no tools (~32 minutes).
- For OLMo-core SFT, we have two test scripts:
  - `scripts/train/debug/oc_sft.sh`: single GPU on Beaker.
  - `scripts/train/debug/oc_sft_multinode.sh`: two 8x GPU nodes on Beaker.
- For DPO, we have three test scripts:
  - `scripts/train/debug/dpo/local.sh`: local single GPU (no Beaker).
  - `scripts/train/debug/dpo/single_gpu.sh`: single GPU on Beaker.
  - `scripts/train/debug/dpo/multi_node.sh`: two 8x GPU nodes on Beaker.
- To run the `./scripts/train/build_image_and_launch.sh` script, you must commit the current changes.
- Launch tool use experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/tools/olmo_3_parser_multigpu.sh`.
- Launch multi-node non-tool experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/large_test_script.sh`.
- Launch OLMo-core SFT experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/oc_sft.sh`.
- Launch multi-node OLMo-core SFT experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/oc_sft_multinode.sh`.
- Launch DPO experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/dpo/single_gpu.sh`.
- Launch multi-node DPO experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/dpo/multi_node.sh`.
- Launch the GPU tests with `./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh`.
- When creating a PR that includes GPU test results, include `GPU_TESTS=[EXPERIMENT_ID](https://beaker.org/ex/EXPERIMENT_ID)` in the PR body. The CI will verify the experiment passed instead of re-running the tests. Use `GPU_TESTS=bypass` to skip GPU tests entirely. **IMPORTANT**: The experiment ID must be from actually running the GPU test script (`scripts/test/run_gpu_pytest.sh`), NOT from training or debug scripts. Training experiments and GPU tests are different things.
- If you are given a Beaker URL (beaker\.allen\.ai.*) use the Beaker CLI tool to interact with it.
- Experiment launch scripts that call `mason.py` must include `--no_auto_dataset_cache` (before the `--` separator) because vllm is not installed locally on macOS. Without this flag, mason.py tries to cache the dataset locally which fails on the `import vllm` in `data_loader.py`.
- The `oe-eval-internal` directory is required in the Docker image for experiments that use `--try_launch_beaker_eval_jobs_on_weka`. If it's missing (e.g. in a fresh clone or worktree), clone it with: `git clone --depth=1 https://github.com/allenai/oe-eval-internal.git oe-eval-internal`.
- When updating PR bodies with experiment results, use the "Runs:" format (numbered list with Beaker links):
  ```
  Runs:

  1. Description: [Beaker](https://beaker.org/ex/EXPERIMENT_ID)
  2. Description: [Beaker](https://beaker.org/ex/EXPERIMENT_ID)
  ```

# Naming conventions
- Models OLMo and OLMo 2 (versions <=2) use the "OLMo" capitalization style.
- Olmo 3, Olmo Hybrid, and later models use "Olmo" (standard proper noun capitalization).
- Note: "OLMo-core" refers to the software repository and keeps its original capitalization.

# Coding conventions
- Never use `import logging` or `logging.info()` directly. Always use `logger = logger_utils.setup_logger(__name__)` and `logger.info()`.
- Imports always go at the top of the file, never inline.
- Use `from package import module` instead of `import package.module`.

# Documentation
To verify that documentation changes don't alter the generated output:
1. Build docs on your branch: `uv run mkdocs build && cp -r site site-branch`
2. Switch to main branch and build: `cd /path/to/main && uv run mkdocs build`
3. Compare the builds: `diff -rq site-branch /path/to/main/site`
4. If no output, the docs are identical. If differences exist, review with: `diff -r site-branch /path/to/main/site`

<!-- edullm:begin -->
<!-- Managed by edu-llm/platform. Edit skills/agents-md-block.md there and re-run
     tools/distribute_agent_layer.py; an edit made here is reverted and, until it is,
     tests/test_agent_layer_is_distributed.py is red. Text outside the markers is
     this repository's own and is never touched. -->

## Running anything on a GPU: use `edullm`, never AWS

This codebase is registered with the eduLLM platform. `edullm` is the only supported way to
reach the cluster from a laptop, and it holds no cloud credential of its own: every AWS
credential lives in a workflow whose trust policy pins it to one file on `main`.

**Do not write a script that calls AWS.** No `boto3`, no `aws` CLI, no `curl` at an AWS
endpoint. For the people here who hold no AWS role that fails, and for the few who do it
succeeds and leaves no run anybody can cite, which is the worse of the two.

```bash
uv tool install --force git+https://github.com/edu-llm/platform
edullm --version
```

Unpinned on purpose, and **re-running that line is the upgrade**. Do not reach for `pip` or
`pipx`, and do not reach for `uv tool upgrade`: what it does depends on how the tool was
installed, so from a release note's pinned line it answers `Nothing to upgrade` and exits 0
however far behind the install is. `uv tool install edullm` is the other near miss and uv
answers `not found in the package registry`, because nothing is published to an index under
that name; the line above installs from git.

If this machine installed before the distribution was renamed, run `uv tool uninstall
edullm-platform` **before** the install line and not after. Both own the same `edullm`
executable and uv deletes it along with the old entry, which leaves a healthy-looking
`uv tool list` and no command.

It needs `gh` logged in and a clone with an `origin` remote, and nothing else — no AWS
profile, no SSO session and no VPN, for anything on this path.

| Verb | What it does |
| --- | --- |
| `edullm check` | Prices a submission from this working tree and lists every refusal. Reaches no network. |
| `edullm submit` | Runs those checks and dispatches the submission workflow. |
| `edullm status` | Names your recent submissions, or describes one run. |
| `edullm logs` | The last lines one run printed. |
| `edullm cancel` | Stops one admitted run, with a reason that goes on the record. |
| `edullm add` | Teaches the platform about a repository, dataset, shape, model or person. |
| `edullm ask` | Files one ask for something you need yourself. |
| `edullm run` / `edullm shell` | Ships this tree to a machine of your own. Ungated, and no run anybody can cite. |

`edullm <verb> --help` prints what that verb takes. The last two are the exploration route
and not the submission path: nothing on them is checked, priced, approved or recorded.

**Start with `edullm check --json`.** It costs a fraction of a second, reaches no network and
lists every refusal at once. Read the JSON on stdout rather than the prose, and **match on
`code`** — the `detail` beside it is written for a person and gets reworded. Exit 0 stands,
1 is refused on the merits, 2 means the command or the install is wrong, 3 means the platform
could not be asked and is the only one worth retrying.

**Never quote a price, a runtime bound, a cost ceiling or who has to approve something from
memory or from a document, this one included.** Those live in reviewed configuration that
changes without anybody being told. Run `edullm check --json` and read `cost` and
`approval_class` out of the output.

Two skills carry the detail and your agent should reach for them by name rather than working
it out here: **submitting-a-run** for anything that runs, and **registering-a-repository**
when `check` refuses with `unregistered_repository`.

Also never: pass `--force` to get past a refusal, edit `.edullm/run.yaml` to silence a
refusal without reading what it says, or commit a secret into this repository — the image is
built from the commit.
<!-- edullm:end -->
