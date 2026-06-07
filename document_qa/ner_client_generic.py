import os
import time

import yaml

"""
This client is a generic client for any Grobid application and sub-modules.
At the moment, it supports only single document processing.

Source: https://github.com/kermitt2/grobid-client-python 
"""

""" Generic API Client """
from copy import deepcopy
import json
import requests

try:
    from urlparse import urljoin
except ImportError:
    from urllib.parse import urljoin


class ApiClient(object):
    """Client to interact with a generic Rest API.

    Subclasses should implement functionality accordingly with the provided
    service methods, i.e. ``get``, ``post``, ``put`` and ``delete``.
    """

    accept_type = "application/xml"
    api_base = None

    def __init__(self, base_url, username=None, api_key=None, status_endpoint=None, timeout=60):
        """Initialise client.

        Args:
            base_url (str): The base URL to the service being used.
            username (str): The username to authenticate with.
            api_key (str): The API key to authenticate with.
            timeout (int): Maximum time before timing out.
        """
        self.base_url = base_url
        self.username = username
        self.api_key = api_key
        self.status_endpoint = urljoin(self.base_url, status_endpoint)
        self.timeout = timeout

    @staticmethod
    def encode(request, data):
        """Add request content data to request body, set Content-type header.

        Should be overridden by subclasses if not using JSON encoding.

        Args:
            request (HTTPRequest): The request object.
            data (dict, None): Data to be encoded.

        Returns:
            HTTPRequest: The request object.
        """
        if data is None:
            return request

        request.add_header("Content-Type", "application/json")
        request.extracted_data = json.dumps(data)

        return request

    @staticmethod
    def decode(response):
        """Decode the returned data in the response.

        Should be overridden by subclasses if something else than JSON is
        expected.

        Args:
            response (HTTPResponse): The response object.

        Returns:
            dict or None.
        """
        try:
            return response.json()
        except ValueError as e:
            return e.message

    def get_credentials(self):
        """Returns parameters to be added to authenticate the request.

        This lives on its own to make it easier to re-implement it if needed.

        Returns:
            dict: A dictionary containing the credentials.
        """
        return {"username": self.username, "api_key": self.api_key}

    def call_api(
        self,
        method,
        url,
        headers=None,
        params=None,
        data=None,
        files=None,
        timeout=None,
    ):
        """Call API.

        This returns object containing data, with error details if applicable.

        Args:
            method (str): The HTTP method to use.
            url (str): Resource location relative to the base URL.
            headers (dict or None): Extra request headers to set.
            params (dict or None): Query-string parameters.
            data (dict or None): Request body contents for POST or PUT requests.
            files (dict or None: Files to be passed to the request.
            timeout (int): Maximum time before timing out.

        Returns:
            ResultParser or ErrorParser.
        """
        headers = deepcopy(headers) or {}
        headers["Accept"] = self.accept_type if "Accept" not in headers else headers["Accept"]
        params = deepcopy(params) or {}
        data = data or {}
        files = files or {}
        # if self.username is not None and self.api_key is not None:
        #    params.update(self.get_credentials())
        r = requests.request(
            method,
            url,
            headers=headers,
            params=params,
            files=files,
            data=data,
            timeout=timeout,
        )

        return r, r.status_code

    def get(self, url, params=None, **kwargs):
        """Call the API with a GET request.

        Args:
            url (str): Resource location relative to the base URL.
            params (dict or None): Query-string parameters.

        Returns:
            ResultParser or ErrorParser.
        """
        return self.call_api("GET", url, params=params, **kwargs)

    def delete(self, url, params=None, **kwargs):
        """Call the API with a DELETE request.

        Args:
            url (str): Resource location relative to the base URL.
            params (dict or None): Query-string parameters.

        Returns:
            ResultParser or ErrorParser.
        """
        return self.call_api("DELETE", url, params=params, **kwargs)

    def put(self, url, params=None, data=None, files=None, **kwargs):
        """Call the API with a PUT request.

        Args:
            url (str): Resource location relative to the base URL.
            params (dict or None): Query-string parameters.
            data (dict or None): Request body contents.
            files (dict or None: Files to be passed to the request.

        Returns:
            An instance of ResultParser or ErrorParser.
        """
        return self.call_api("PUT", url, params=params, data=data, files=files, **kwargs)

    def post(self, url, params=None, data=None, files=None, **kwargs):
        """Call the API with a POST request.

        Args:
            url (str): Resource location relative to the base URL.
            params (dict or None): Query-string parameters.
            data (dict or None): Request body contents.
            files (dict or None: Files to be passed to the request.

        Returns:
            An instance of ResultParser or ErrorParser.
        """
        return self.call_api(method="POST", url=url, params=params, data=data, files=files, **kwargs)

    def service_status(self, **kwargs):
        """Call the API to get the status of the service.

        Returns:
            An instance of ResultParser or ErrorParser.
        """
        return self.call_api("GET", self.status_endpoint, params={"format": "json"}, **kwargs)


class NERClientGeneric(ApiClient):
    def __init__(self, config_path=None, ping=False):
        self.config = None
        if config_path is not None:
            self.config = self._load_yaml_config_from_file(path=config_path)
            super().__init__(self.config["grobid"]["server"])

            if ping:
                result = self.ping_service()
                if not result:
                    raise Exception("Grobid is down.")

        os.environ["NO_PROXY"] = "nims.go.jp"

    @staticmethod
    def _load_json_config_from_file(path="./config.json"):
        """
        Load the json configuration
        """
        config = {}
        with open(path, "r") as fp:
            config = json.load(fp)

        return config

    @staticmethod
    def _load_yaml_config_from_file(path="./config.yaml"):
        """
        Load the YAML configuration
        """
        config = {}
        try:
            with open(path, "r") as the_file:
                raw_configuration = the_file.read()

            config = yaml.safe_load(raw_configuration)
        except Exception as e:
            print("Configuration could not be loaded: ", str(e))
            exit(1)

        return config

    def set_config(self, config, ping=False):
        self.config = config
        if ping:
            try:
                result = self.ping_service()
                if not result:
                    raise Exception("Grobid is down.")
            except Exception as e:
                raise Exception("Grobid is down or other problems were encountered. ", e)

    def ping_service(self):
        # test if the server is up and running...
        ping_url = self.get_url("ping")

        r = requests.get(ping_url)
        status = r.status_code

        if status != 200:
            print("GROBID server does not appear up and running " + str(status))
            return False
        else:
            print("GROBID server is up and running")
            return True

    def get_url(self, action):
        grobid_config = self.config["grobid"]
        base_url = grobid_config["server"]
        action_url = base_url + grobid_config["url_mapping"][action]

        return action_url

    def process_texts(self, input, method_name="superconductors", params={}, headers={"Accept": "application/json"}):

        files = {"texts": input}

        the_url = self.get_url(method_name)
        params, the_url = self.get_params_from_url(the_url)

        res, status = self.post(url=the_url, files=files, data=params, headers=headers)

        if status == 503:
            time.sleep(self.config["sleep_time"])
            return self.process_texts(input, method_name, params, headers)
        elif status != 200:
            print("Processing failed with error " + str(status))
            return status, None
        else:
            return status, json.loads(res.text)

    def process_text(self, input, method_name="superconductors", params={}, headers={"Accept": "application/json"}):

        files = {"text": input}

        the_url = self.get_url(method_name)
        params, the_url = self.get_params_from_url(the_url)

        res, status = self.post(url=the_url, files=files, data=params, headers=headers)

        if status == 503:
            time.sleep(self.config["sleep_time"])
            return self.process_text(input, method_name, params, headers)
        elif status != 200:
            print("Processing failed with error " + str(status))
            return status, None
        else:
            return status, json.loads(res.text)

    def process_pdf(self, form_data: dict, method_name="superconductors", params={}, headers={"Accept": "application/json"}):

        the_url = self.get_url(method_name)
        params, the_url = self.get_params_from_url(the_url)

        res, status = self.post(url=the_url, files=form_data, data=params, headers=headers)

        if status == 503:
            time.sleep(self.config["sleep_time"])
            return self.process_text(input, method_name, params, headers)
        elif status != 200:
            print("Processing failed with error " + str(status))
        else:
            return res.text

    def process_pdfs(self, pdf_files, params={}):
        pass

    def process_pdf(self, pdf_file, method_name, params={}, headers={"Accept": "application/json"}, verbose=False, retry=None):

        files = {"input": (pdf_file, open(pdf_file, "rb"), "application/pdf", {"Expires": "0"})}

        the_url = self.get_url(method_name)

        params, the_url = self.get_params_from_url(the_url)

        res, status = self.post(url=the_url, files=files, data=params, headers=headers)

        if status == 503 or status == 429:
            if retry is None:
                retry = self.config["max_retry"] - 1
            else:
                if retry - 1 == 0:
                    if verbose:
                        print("re-try exhausted. Aborting request")
                    return None, status
                else:
                    retry -= 1

            sleep_time = self.config["sleep_time"]
            if verbose:
                print("Server is saturated, waiting", sleep_time, "seconds and trying again. ")
            time.sleep(sleep_time)
            return self.process_pdf(pdf_file, method_name, params, headers, verbose=verbose, retry=retry)
        elif status != 200:
            desc = None
            if res.content:
                c = json.loads(res.text)
                desc = c["description"] if "description" in c else None
            return desc, status
        elif status == 204:
            # print('No content returned. Moving on. ')
            return None, status
        else:
            return res.text, status

    def get_params_from_url(self, the_url):
        """
        This method is used to pass to the URL predefined parameters, which are added in the URL format
        """
        params = {}
        if "?" in the_url:
            split = the_url.split("?")
            the_url = split[0]
            params = split[1]

            params = {param.split("=")[0]: param.split("=")[1] for param in params.split("&")}
        return params, the_url
