from api.api import API_dtypes, API_rtypes, APIEndpoint, APISpec

CLOUD_BASIC_SPEC = APISpec()

CLOUD_BASIC_SPEC.add_endpoint(APIEndpoint(
    name="request",
    route="/request",
    rtype=API_rtypes.POST,

    description="Endpoint for text generation",
    input_type=API_dtypes.TEXT,
    output_type=API_dtypes.JSON,
    possible_returns={
        200: "Ok",
        403: "Incorrect token"
    },
))
