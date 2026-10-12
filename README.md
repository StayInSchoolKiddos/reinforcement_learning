# Installing

## Host networking

All Compose services use `network_mode: host`. On Linux, they share the host's
network stack, so no Docker port mappings are needed:

- Ray dashboard: <http://localhost:8265> while Ray is running
- JupyterLab: <http://localhost:8888>
- TensorBoard: <http://localhost:6006>

Apply networking changes to existing containers with `docker compose up -d`.
This recreates affected containers and interrupts any running training or
notebook kernels; rebuilding the image is not required.

Host networking removes network isolation. Services bound to `0.0.0.0` are
reachable from other machines if the host firewall allows it. Keep Ray
authentication enabled and set a private `JUPYTER_TOKEN` in your local `.env`
instead of using the default token on a shared network.

Only one process can bind each host port. Do not run CPU and GPU Ray dashboards
on port 8265 simultaneously. Host networking does not share IPC or `/dev/shm`;
the configured `SHM_SIZE` still controls each container's shared memory.

## Ray dashboard

Run `python3 src/train_ppo.py` inside the `rllib` container. The script starts
Ray's dashboard on all interfaces at host port 8265. Open
<http://localhost:8265> while training is running. If Docker
is on a remote host, forward port 8265 through VS Code's Ports panel or SSH.

Ray 2.59 enables token authentication by default. If the dashboard asks for a
token, retrieve it inside the same container with `ray get-auth-token`; do not
share the token. Keep authentication enabled when exposing the dashboard.

The dashboard belongs to the script's local Ray instance and stops when the
Python process exits, including when you press Ctrl+C. Increase `ITERS` for a
longer run if needed.

If Ray warns that it is using `/tmp/ray` instead of `/dev/shm`, increase
`SHM_SIZE` in your local `.env` to at least the size Ray recommends (for example,
`SHM_SIZE=6gb` for a recommendation of 4.80 GB), subject to available host memory,
then recreate the affected container. This is a performance warning, not the
cause of an inaccessible dashboard.

## TensorBoard learning curves

The PPO script writes mean episode return after every training iteration to a
separate run under `results/tensorboard/ppo_cartpole_*`. The TensorBoard service
reads `results/`, including these event logs. Rerun the updated script to generate
them; existing checkpoint files cannot supply past learning curves.

Open <http://localhost:6006> and select Scalars, then
`env_runners/episode_return_mean`. Logs remain available after training exits.
If VS Code is connected to a remote host, manually forward port 6006 in the
Ports panel and use its Open in Browser action; it need not appear automatically.

If the page is unreachable, check `docker compose ps tensorboard` and
`docker compose logs --tail 30 tensorboard`. Start the service with
`docker compose up -d tensorboard` if needed. TensorBoard does not use the Ray
authentication token.