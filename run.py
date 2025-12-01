# import depth_tools
# import denoising_engine
import argparse
import os
import sys


def parse_arguments(args):
    usage_text = "Usage:  python run.py [options]"
    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument(
        "-i", "--input_dir", type=str, help="Input directory.", required=True
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, help="Output directory.", required=True
    )
    parser.add_argument(
        "-f",
        "--flying_remove",
        action="store_true",
        help="Where to remove flying pixels.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.23,
        help="Threshold for flying pixels.",
    )
    parser.add_argument("-s1", "--skip1", action="store_true", help="Skip stage 1.")
    parser.add_argument("-s2", "--skip2", action="store_true", help="Skip stage 2.")
    parser.add_argument(
        "-n", "--number", type=int, default=2, help="Number of iterations in stage 2."
    )
    parser.add_argument("--alpha", type=float, help="Alpha for stage 2.", default=1.0)
    parser.add_argument("--beta", type=float, help="Beta for stage 2.", default=0.5)
    return parser.parse_known_args(args)


def denoise_dir(args):
    cam_list_noisy = depth_tools.load_depth_scene(args.input_dir)

    denoiser = denoising_engine.Denoiser(verbose=True)

    denoiser.initialize()

    if args.flying_remove:
        cam_list_noisy = denoiser.remove_flying(
            cam_list_noisy, threshold=args.threshold
        )

    cam_list_out, dt1, dt2 = denoiser.denoise_multiview(
        cam_list_noisy, steps_stage2=args.number, alpha=args.alpha, beta=args.beta
    )

    print(f"[RUNTIME={dt1 + dt2:.3f}]")

    for cam in cam_list_out:
        cam.save(args.output_dir)

    pcl_list = [cam.get_pcl() for cam in cam_list_out]
    pcl = depth_tools.merge_pcl(pcl_list)
    pcl.write_ply(os.path.join(args.output_dir, "cloud.ply"))


def main():
    args, _ = parse_arguments(sys.argv)
    denoise_dir(args)


if __name__ == "__main__":
    main()
