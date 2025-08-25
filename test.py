import qci_client as qc

# the following line shows how to directly configure QCIClient with variables
# client = qc.QciClient(api_token=<your_secret_token>, url="https://api.qci-prod.com")
# if you have configured your environment correctly, the following line should work
client = qc.QciClient()

print(client.get_allocations()["allocations"]["dirac"])