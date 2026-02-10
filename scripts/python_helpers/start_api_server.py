import argparse
import os
import shlex

import govscape as gs


def _get_arg_parser():
    parser = argparse.ArgumentParser(description="Start the GovScape API server")
    parser.add_argument(
        "-d",
        "--data-directory",
        default="data/test_data",
        help="Directory containing data files",
    )
    parser.add_argument(
        "-tm", "--text_model", default="ST", help="The model to use for text embedding."
    )
    parser.add_argument(
        "-vm",
        "--visual_model",
        default="CLIP",
        help="The model to use for visual embedding.",
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=20, help="Number of top results to return"
    )
    parser.add_argument(
        "-i",
        "--vector_index_type",
        default="Memory",
        help="The type of vector index to use",
    )
    parser.add_argument(
        "-ki",
        "--keyword_index_type",
        default="LanceDB",
        help="The type of keyword index to use",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to run the server on"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run the server in debug mode"
    )
    return parser


def _build_app_from_args(args):
    if args.text_model == "ST":
        text_model = gs.ST_TextEmbeddingModel()
    elif args.text_model == "BGE":
        text_model = gs.BGE_TextEmbeddingModel()
    elif args.text_model == "BGESmall":
        text_model = gs.BGESmall_TextEmbeddingModel()
    elif args.text_model == "Dummy":
        text_model = gs.Dummy_TextEmbeddingModel()
    else:
        raise ValueError(f"Unsupported text model: {args.text_model}")

    if args.visual_model == "CLIP":
        visual_model = gs.CLIP_VisualEmbeddingModel()
    elif args.visual_model == "Dummy":
        visual_model = gs.Dummy_VisualEmbeddingModel()
    else:
        raise ValueError(f"Unsupported visual model: {args.visual_model}")

    index_config = gs.IndexConfig(
        args.data_directory, args.vector_index_type, args.keyword_index_type
    )

    server_config = gs.ServerConfig(
        index_config, text_model, visual_model, k=args.top_k
    )
    return gs.Server(server_config)


def create_app():
    parser = _get_arg_parser()
    app_args = os.getenv("APP_ARGS", "")
    if app_args:
        args = parser.parse_args(shlex.split(app_args))
    else:
        args = parser.parse_args([])
    return _build_app_from_args(args).app


def main():
    parser = _get_arg_parser()
    args = parser.parse_args()
    server = _build_app_from_args(args)
    server.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
