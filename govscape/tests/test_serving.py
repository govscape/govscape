import govscape as gs
import argparse
import pytest
import os


def check_returned_results(file, server):
    with open(file, 'r', encoding='utf-8') as f:
            pdf = f.readline()
            correct = 0
            while pdf:
                for _ in range(0, 10):
                    search = f.readline()
                    print(search)

                    results = server.search(search)["results"]
                    print(results)

                    for result in results:
                        if pdf in result["pdf"]:
                            correct += 1
                print(pdf + " " + str(correct))
                pdf = f.readline()
                correct = 0

def test_server():
    data_directory='tests/test_data/large'
    model_type='ST'
    index_type = "Memory"
    txt_directory = os.path.join(data_directory, 'txt')
    embeddings_directory = os.path.join(data_directory, 'embeddings')
    index_directory = os.path.join(data_directory, 'index')
    image_directory = os.path.join(data_directory, 'images')

    if model_type == "CLIP":
        model = gs.CLIPEmbeddingModel()
    elif model_type == "ST":
        model = gs.ST_TextEmbeddingModel()
    else:
        raise ValueError("Unsupported model type")

    index_config = gs.IndexConfig(embeddings_directory, index_directory, image_directory, index_type)
    server_config = gs.ServerConfig(index_config, gs.PDFsToEmbeddings(txt_directory, embeddings_directory, image_directory, model), k = 5)
    s = gs.Server(server_config)
    check_returned_results('tests/test_data/large/queries/test.txt', s)