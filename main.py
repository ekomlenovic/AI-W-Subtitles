from SubtitleGenerator import SubtitleGenerator
import argparse


def main():
    """Main function to handle command-line arguments and process the video."""
    parser = argparse.ArgumentParser(description="Video Subtitle Generator")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--model_name", default="turbo", choices=["tiny", "base", "small", "medium", "large", "turbo"], help="Whisper model name")
    parser.add_argument("--translate", choices=["fr", "en"], help="Language to translate subtitles to (fr or en)")

    args = parser.parse_args()

    print(f"Processing video: {args.video_path}")
    print(f"Whisper model: {args.model_name}")
    if args.translate:
        print(f"Translating subtitles to: {args.translate}")
    generator = SubtitleGenerator(args.model_name)
    generator.process_video(args.video_path, args.translate)

if __name__ == "__main__":
    main()
