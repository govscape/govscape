import json

from .config import ServerConfig


class Filter:
    def __init__(self, config: ServerConfig):
        self.embedding_directory = config.embedding_directory

    # inputs: file_json to get metadata from, f = name of field we want data
    # outputs the data value of f
    def json_get_data(self, file_json, f):
        with open(file_json) as file:
            data = json.load(file)
        return data.get(f)

    # inputs: search_results = list of search results from server
    # filters = dict of filters to consider. key = filter, val = tuple that
    #   indicates range
    # returns: search_results refined after the filters
    def filter_results(self, search_results, filters):
        filtered_results = []
        for f in filters:
            for sr in search_results:
                # get filter info from json with that specific filename.
                filename = sr["pdf"] + ".json"
                f_val = self.json_get_data(filename, f)
                # check if within filter
                if f == "timestamp":  # TODO: when add month, day functionality
                    if f == "timestamp":
                        f_val = int(f_val[:4])
                    if filters[f][0] == filters[f][1]:
                        if f_val == filters[f][0]:
                            filtered_results.append(sr)
                    else:
                        if filters[f][0] <= f_val <= filters[f][1]:
                            filtered_results.append(sr)
                elif f == "num_pages":
                    if filters[f][0] == filters[f][1]:
                        if f_val == filters[f][0]:
                            filtered_results.append(sr)
                    else:
                        if filters[f][0] <= f_val <= filters[f][1]:
                            filtered_results.append(sr)
                else:  # then it must be government name.
                    if filters[f] == f_val:
                        filtered_results.append(sr)
            search_results = filtered_results
        return filtered_results
