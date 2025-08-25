import qci_client as qc

# Objective function:
# f(x1, x2, x3, x4) =
#   0.760885 * x1**2 + 0.7174 * x2**2 + 0.79094 * x3**2 + 0.79094 * x4**2
#   + 0.12774 * x1*x2 + 0.64883 * x1*x3 + 0.62263 * x1*x4
#   + 0.32937 * x2*x3 + 0.3374 * x2*x4 + 1.2294 * x3*x4
#   - 0.59754 * x1 - 0.31987 * x2 - 1.32466 * x3 - 1.3784 * x4


client = qc.QciClient()


poly_coefficients = [0.760885, 0.7174, 0.79094, 0.79094,
                      0.12774, 0.64883, 0.62263, 0.32937,
                        0.3374, 1.2294, -0.59754, -0.31987,
                          -1.32466, -1.3784]

poly_indices = [[1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [1, 2], 
                [1, 3], 
                [1, 4],
                [2, 3], 
                [2, 4], 
                [3, 4],
                [0, 1], 
                [0, 2], 
                [0, 3], 
                [0, 4]]


data = []
for i in range(len(poly_coefficients)):
    data.append({
        "val": poly_coefficients[i],
        "idx": poly_indices[i]
    })
poly_file = {"file_name": "test-polynomial",
             "file_config": {"polynomial": {
                 "min_degree": 1,
                 "max_degree": 2,
                 "num_variables":4,
                 "data": data
             }}}
file_id = client.upload_file(file=poly_file)["file_id"]


job_body = client.build_job_body(job_type="sample-hamiltonian", polynomial_file_id=file_id, job_params={"device_type": "dirac-3", "sum_constraint": 1.58188 , "relaxation_schedule": 1, "num_samples": 15})

response = client.process_job(job_body=job_body)


print (response)