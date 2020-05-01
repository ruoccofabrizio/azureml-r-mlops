library(azuremlsdk)
library(magrittr)
library(optparse)

options <- list(
  make_option("--tenant_id", default=""),
  make_option("--service_principal_id", default=""),
  make_option("--service_principal_password", default=""),
  make_option("--aml_workspace", default=""),
  make_option("--aml_subscription_id", default=""),
  make_option("--aml_resource_group", default=""),
  make_option("--aml_cluster", default="")
)

# Parse option from command line
opt_parser <- OptionParser(option_list = options)
opt <- parse_args(opt_parser)

# Create a Service Principal Credential to connect to the Azure ML workspace
svc_pr <- service_principal_authentication(opt$tenant_id, opt$service_principal_id,
  opt$service_principal_password, cloud = "AzureCloud")

# Get AML Workspace
ws <- get_workspace(opt$aml_workspace,
                    opt$aml_subscription_id,
                    opt$aml_resource_group,
                    auth = svc_pr)
print(ws)

# Define experiment
experiment_name <- "accident-logreg-mlops"
exp <- experiment(ws, experiment_name)
print(exp)

# Get or create cluster
cluster_name <- opt$aml_cluster
compute_target <- get_compute(ws, cluster_name = cluster_name)
if (is.null(compute_target)) {
  vm_size <- "STANDARD_D2_V2" 
  compute_target <- create_aml_compute(workspace = ws,
                                       cluster_name = cluster_name,
                                       vm_size = vm_size,
                                       max_nodes = 2)
  
  wait_for_provisioning_completion(compute_target, show_output = TRUE)
}
print(compute_target)


## Prepare data for training
nassCDS <- read.csv("./code/nassCDS.csv", 
                     colClasses=c("factor","numeric","factor",
                                  "factor","factor","numeric",
                                  "factor","numeric","numeric",
                                  "numeric","character","character",
                                  "numeric","numeric","character"))

accidents <- na.omit(nassCDS[,c("dead","dvcat","seatbelt","frontal","sex","ageOFocc","yearVeh","airbag","occRole")])
accidents$frontal <- factor(accidents$frontal, labels=c("notfrontal","frontal"))
accidents$occRole <- factor(accidents$occRole)
accidents$dvcat <- ordered(accidents$dvcat, 
                          levels=c("1-9km/h","10-24","25-39","40-54","55+"))

saveRDS(accidents, file="accidents.Rd")

### Upload data to the datastore
# Get default datastore
ds <- get_default_datastore(ws)
# Define path and upload
target_path <- "accidentdata"
upload_files_to_datastore(ds,
                          list("./accidents.Rd"),
                          target_path = target_path,
                          overwrite = TRUE)


## Train a model
# Define estimator
est <- estimator(source_directory = ".",
                 entry_script = "./code/accidents.R",
                 script_params = list("--data_folder" = ds$path(target_path)),
                 compute_target = compute_target
                 )
print(est)

### Submit the job on the remote cluster
run <- submit_experiment(exp, est)

wait_for_run_completion(run, show_output = TRUE)

# Get metrics
metrics <- get_run_metrics(run)
metrics

# Get the trained model
download_files_from_run(run, prefix="outputs/")
accident_model <- readRDS("outputs/model.rds")
summary(accident_model)


# Register the model
model <- register_model(ws, 
                        model_path = "outputs/model.rds", 
                        model_name = "accidents_model",
                        description = "Predict probablity of auto accident")

# Deployment will be part of deploy.R script. Commented out here
# # Create environment
# r_env <- r_environment(name = "basic_env")

# # Create inference config
# inference_config <- inference_config(
#   entry_script = "accident_predict.R",
#   source_directory = ".",
#   environment = r_env)

# #Deploy to ACI
# aci_config <- aci_webservice_deployment_config(cpu_cores = 1, memory_gb = 0.5)

# aci_service <- deploy_model(ws, 
#                             'accident-pred', 
#                             list(model), 
#                             inference_config, 
#                             aci_config)

# wait_for_deployment(aci_service, show_output = TRUE)