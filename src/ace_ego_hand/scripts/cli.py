"""The `ace-ego-hand` console command: one tyro subcommand per script.

ace-ego-hand infer clip.mp4 --render
ace-ego-hand visualize clip.mp4
ace-ego-hand lerobot /path/to/dataset --video-key observation.images.front
ace-ego-hand undistort fisheye.mp4 --camera fisheye.camera.parquet
ace-ego-hand demo
"""

from __future__ import annotations

from typing import Annotated

import tyro

from . import demo, infer, lerobot, undistort, visualize


def main() -> None:
    command = tyro.cli(
        tyro.conf.OmitSubcommandPrefixes[
            Annotated[infer.InferConfig, tyro.conf.subcommand("infer")]
            | Annotated[visualize.VisualizeConfig, tyro.conf.subcommand("visualize")]
            | Annotated[lerobot.LerobotConfig, tyro.conf.subcommand("lerobot")]
            | Annotated[undistort.UndistortConfig, tyro.conf.subcommand("undistort")]
            | Annotated[demo.DemoConfig, tyro.conf.subcommand("demo")]
        ],
        description="ACE-Ego-Hand: bimanual 3D hand motion from egocentric video.",
    )
    match command:
        case infer.InferConfig():
            infer.main(command)
        case visualize.VisualizeConfig():
            visualize.main(command)
        case lerobot.LerobotConfig():
            lerobot.main(command)
        case undistort.UndistortConfig():
            undistort.main(command)
        case demo.DemoConfig():
            demo.main(command)


if __name__ == "__main__":
    main()
