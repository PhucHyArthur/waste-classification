import json
import xmltodict


def xml2dict(xml_path: str,
             saved_dict_path: str=None) -> dict:
    """"""
    with open(xml_path) as xml_file:
        data_dict = xmltodict.parse(xml_file.read())

        if saved_dict_path:
            json_data = json.dumps(data_dict)

            with open(saved_dict_path, "w") as json_file:
                json_file.write(json_data)

    return data_dict
