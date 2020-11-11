import logging

from azureml.core import Workspace, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import ComputeTarget, AmlCompute
from dask_cloudprovider import AzureMLCluster
import click, os, time, re

from azureml_ngc_tools.AzureMLComputeCluster import AzureMLComputeCluster
from azureml_ngc_tools.cli import ngccontent
from azureml.exceptions._azureml_exception import ProjectSystemException

### SETUP LOGGING
fileFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.10s]  %(message)s")
logger = logging.getLogger("azureml_ngc")
logger.setLevel("DEBUG")

fileHandler = logging.FileHandler("azureml_ngc_tools.log", mode="w")
fileHandler.setFormatter(fileFormatter)
logger.addHandler(fileHandler)


@click.command()
@click.option(
    "--login", is_flag=False, required=True, help="Path to the login config file"
)
@click.option("--app", is_flag=False, required=True, help="Path to the config file")
@click.version_option()
def start(login, app):
    login_config = ngccontent.get_config(login)
    app_config = ngccontent.get_config(app)

    ### WORKSPACE
    subscription_id = login_config["azureml_user"]["subscription_id"]
    resource_group = login_config["azureml_user"]["resource_group"]
    workspace_name = login_config["azureml_user"]["workspace_name"]

    try:
        ws = Workspace(
            workspace_name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group,
        )
    except ProjectSystemException:
        msg = f'\n\nThe workspace "{workspace_name}" does not exist. '
        msg += f"Go to \n\n  "
        msg += f"-->> https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace <<--\n\n"
        msg += f"and create the workspace first.\n\n\n"
        msg += f"Your current configuration: \n\n"
        msg += f"Workspace name: {workspace_name} \n"
        msg += f"Subscription id: {subscription_id} \n"
        msg += f"Resource group: {resource_group}\n\n"

        logger.exception(msg)
        raise Exception(msg)

    verify = f"""
    Subscription ID: {subscription_id}
    Resource Group: {resource_group}
    Workspace: {workspace_name}"""
    logger.info(verify)

    ### experiment name
    exp_name = login_config["aml_compute"]["exp_name"]

    ### azure ml names
    ct_name = login_config["aml_compute"]["ct_name"]
    vm_name = login_config["aml_compute"]["vm_name"].lower()
    vm_priority = login_config["aml_compute"]["vm_priority"]

    ### trust but verify
    verify = f"""
    Experiment name: {exp_name}"""
    logger.info(verify)

    ### GPU RUN INFO
    workspace_vm_sizes = AmlCompute.supported_vmsizes(ws)
    pascal_volta_pattern = pattern = re.compile(
        r"[a-z]+_nc[0-9]+[s]?_v[2,3]"
    )  ### matches NC-series v2 and v3
    workspace_vm_sizes = [
        (e["name"].lower(), e["gpus"])
        for e in workspace_vm_sizes
        if pattern.match(e["name"].lower())
    ]
    workspace_vm_sizes = dict(workspace_vm_sizes)

    ### GET COMPUTE TARGET
    if vm_name in workspace_vm_sizes:
        gpus_per_node = workspace_vm_sizes[vm_name]

        verify = f"""
    Compute target: {ct_name}
    VM Size: {vm_name}
    No of GPUs: {gpus_per_node}
    Priority: {vm_priority}
        """
        logger.info(verify)

        ### get SSH keys
        ssh_key_pub, pri_key_file = get_ssh_keys()

        if ct_name not in ws.compute_targets:
            logger.warning(f"Compute target {ct_name} does not exist...")
            ct = createOrGetComputeTarget(
                ws, ct_name, vm_name, vm_priority, ssh_key_pub, login_config
            )
        else:
            ct = ws.compute_targets[ct_name]

            if ct.provisioning_state == "Failed":
                logger.warning(
                    f"Compute target {ct_name} found but provisioning_state is showing as 'failed'..."
                )
                logger.warning(f"Deleting {ct_name} target and will attempt again...")
                logger.warning(
                    f"If this fails again check that you have enough resources in your subscription..."
                )

                ct.delete()
                time.sleep(5)
                ct = createOrGetComputeTarget(
                    ws, ct_name, vm_name, vm_priority, ssh_key_pub, login_config
                )
            else:
                logger.info(f"    Using pre-existing compute target {ct_name}")
    else:
        logger.exception("Unsupported vm_size {vm_size}".format(vm_size=vm_name))
        logger.exception("The specified vm size must be one of ...")

        for azure_gpu_vm_size in workspace_vm_sizes.keys():
            logger.exception("... " + azure_gpu_vm_size)
        raise Exception(
            "{vm_size} does not have Pascal or above GPU Family".format(vm_size=vm_name)
        )

    env = createOrGetEnvironment(ws, login_config, app_config)

    ### UPLOAD ADDITIONAL CONTENT IF NOT EXISTS
    for additional_content in app_config["additional_content"]["list"]:
        url = additional_content["url"]
        targetfile = additional_content["filename"]
        src_path = additional_content["localdirectory"]
        dest_path = additional_content["computedirectory"]
        
        if (
            "source" in additional_content.keys()
            and additional_content["source"] == "github"
        ):
            ngccontent.clone_github_repo(url,"additional_content",src_path)
        else:
            if app_config["additional_content"]["download_content"]:
                ngccontent.download(url, "additional_content", targetfile)
            if (
                app_config["additional_content"]["unzip_content"]
                and additional_content["zipped"]
            ):
                ngccontent.unzipFile(targetfile, "additional_content", src_path)

        if app_config["additional_content"]["upload_content"]:
            ngccontent.upload_data(
                ws,
                ws.get_default_datastore(),
                "additional_content/" + src_path,
                dest_path,
            )

    if (login_config["aml_compute"]["max_nodes"]==1):    
        amlcluster = AzureMLComputeCluster(
            workspace=ws,
            compute_target=ct,
            initial_node_count=1,
            experiment_name=login_config["aml_compute"]["exp_name"],
            environment_definition=env,
            jupyter_port=login_config["aml_compute"]["jupyter_port"],
            telemetry_opt_out=login_config["azureml_user"]["telemetry_opt_out"],
            admin_username=login_config["aml_compute"]["admin_name"],
            admin_ssh_key=pri_key_file,
        )
    else:
        logger.info("Creating a Dask Cluster with {} nodes".format(login_config["aml_compute"]["max_nodes"]))
        amlcluster = AzureMLCluster(
            workspace=ws,
            compute_target=ct,
            initial_node_count=login_config["aml_compute"]["max_nodes"],
            experiment_name=login_config["aml_compute"]["exp_name"],
            environment_definition=env,
            jupyter_port=login_config["aml_compute"]["jupyter_port"],
            telemetry_opt_out=login_config["azureml_user"]["telemetry_opt_out"],
            admin_username=login_config["aml_compute"]["admin_name"],
            admin_ssh_key=pri_key_file,
        )       

    logger.info(f"\n    Go to: {amlcluster.jupyter_link}")
    logger.info("    Press Ctrl+C to stop the cluster.")

    try:
        while True:
            pass
    except KeyboardInterrupt:
        amlcluster.close()


def get_ssh_keys():
    from cryptography.hazmat.primitives import serialization as crypto_serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend as crypto_default_backend

    dir_path = os.path.join(os.getcwd(), ".ssh")

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    pub_key_file = os.path.join(dir_path, "key.pub")
    pri_key_file = os.path.join(dir_path, "key")

    keys_exist = True

    if not os.path.exists(pub_key_file):
        print("Public SSH key does not exist!")
        keys_exist = False

    if not os.path.exists(pri_key_file):
        print("Private SSH key does not exist!")
        keys_exist = False

    if not keys_exist:
        key = rsa.generate_private_key(
            backend=crypto_default_backend(), public_exponent=65537, key_size=2048
        )

        private_key = key.private_bytes(
            crypto_serialization.Encoding.PEM,
            crypto_serialization.PrivateFormat.PKCS8,
            crypto_serialization.NoEncryption(),
        )
        public_key = key.public_key().public_bytes(
            crypto_serialization.Encoding.OpenSSH,
            crypto_serialization.PublicFormat.OpenSSH,
        )

        with open(pub_key_file, "wb") as f:
            f.write(public_key)

        with open(pri_key_file, "wb") as f:
            f.write(private_key)

        os.chmod(pri_key_file, 0o600)

    with open(pub_key_file, "r") as f:
        pubkey = f.read()

    return pubkey, pri_key_file


def createOrGetComputeTarget(
    ws, ct_name, vm_name, vm_priority, ssh_key_pub, login_config
):
    config = AmlCompute.provisioning_configuration(
        vm_size=vm_name,
        min_nodes=login_config["aml_compute"]["min_nodes"],
        max_nodes=login_config["aml_compute"]["max_nodes"],
        vm_priority=vm_priority,
        idle_seconds_before_scaledown=login_config["aml_compute"][
            "idle_seconds_before_scaledown"
        ],
        admin_username=login_config["aml_compute"]["admin_name"],
        admin_user_ssh_key=ssh_key_pub,
        remote_login_port_public_access="Enabled",
    )
    ct = ComputeTarget.create(ws, ct_name, config)
    ct.wait_for_completion(show_output=True)

    if ct.provisioning_state != "Succeeded":
        msg = f"Failed to create the cluster..."
        logger.exception(msg)
        raise Exception(msg)
    return ct


def createOrGetEnvironment(ws, login_config, app_config):
    environment_name = login_config["aml_compute"]["environment_name"]
    python_interpreter = login_config["aml_compute"]["python_interpreter"]
    conda_packages = login_config["aml_compute"]["conda_packages"]

    ### CREATE OR RETRIEVE THE ENVIRONMENT
    if environment_name not in ws.environments:
        logger.info(f"Creating {environment_name} environment...")
        env = Environment(name=environment_name)
        env.docker.enabled = login_config["aml_compute"]["docker_enabled"]
        env.docker.base_image = None
        env.docker.base_dockerfile = f'FROM {app_config["base_dockerfile"]}'
        env.python.interpreter_path = python_interpreter
        env.python.user_managed_dependencies = True
        conda_dep = CondaDependencies()

        for conda_package in conda_packages:
            conda_dep.add_conda_package(conda_package)

        env.python.conda_dependencies = conda_dep
        env.register(workspace=ws)
        evn = env
    else:
        logger.info(f"    Environment {environment_name} found...")
        env = ws.environments[environment_name]

    return env


def go():
    start()


if __name__ == "__main__":
    go()
