import logging
from datetime import timedelta, datetime

import json
import subprocess

from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import BranchPythonOperator

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def check_branch(**kwargs):

    bash_command = [kwargs['python_path'],
                    kwargs['jobs_path'] + "/main.py",
                    "{}".format(kwargs['task_code']),
                    "--account_id",
                    kwargs['account_id'],
                    "--ds_nodash",
                    kwargs['ds_nodash'],
                    "--conf",
                    kwargs['conf_code']]
    logging.debug(" ".join(bash_command))
    code = str(subprocess.call(bash_command))
    logging.debug("Job return code : %d --> task %s", code, kwargs[code])

    return kwargs[code]


class DagBuilder(object):
    CONF_BO   = "/etc/conf/backoffice.json"
    CONF_CODE = "/etc/conf/purchase-probability.yaml"
    PATH_CODE = "/opt/purchase-probability"
    PYTHON_VENV_PATH = "/venv/bin/python"
    PYTHON_JOBS_PATH = "/python"

    def __init__(self):

        with open(DagBuilder.CONF_BO, "r") as json_config:
            self.backoffice_conf = json.loads(json_config.read())

        self.python_path = DagBuilder.PATH_CODE + DagBuilder.PYTHON_VENV_PATH
        self.jobs_path   = DagBuilder.PATH_CODE + DagBuilder.PYTHON_JOBS_PATH

    def __get_accounts(self):

        return []
        # TODO: FROM BACKOFFICE

    @staticmethod
    def create_template_dag(dag_id):
        default_args = {
            'owner': '🎯',
        }

        _dag = DAG(
            dag_id=dag_id,
            schedule_interval=None,
            start_date=datetime(2020, 1, 1),
            default_args=default_args,
            max_active_runs=1,
            catchup=False
        )

        return _dag

    def create_template_task(self, dag_id, account_id, task, trigger_rule='all_success'):

        execute_command = self.python_path + " " + \
                          self.jobs_path + "/main.py" + \
                          " {}".format(task) + \
                          " --account_id {}".format(account_id) + \
                          " --ds_nodash {}".format('{{ ds_nodash }}') + \
                          " --env cloud" + \
                          " --mode prod" + \
                          " --conf {}".format(DagBuilder.CONF_CODE)

        execute = BashOperator(
            task_id=task,
            bash_command=execute_command,
            retries=0,
            retry_delay=timedelta(minutes=10),
            dag=globals()[dag_id],
            trigger_rule=trigger_rule)

        return execute

    def create_account_dags(self):
        accounts = self.__get_accounts()
        template_dag_id = "purchase_probability_{}"

        for account_id in accounts:

            dag_id = template_dag_id.format(account_id)

            globals()[dag_id] = self.create_template_dag(dag_id)

            # TASKS

            task_start = DummyOperator(task_id="start", dag=globals()[dag_id])

            task_check_model = BranchPythonOperator(task_id="check_model",
                                                    python_callable=check_branch,
                                                    op_kwargs={'account_id': account_id,
                                                               'task_code': 'check_model',
                                                               'python_path': self.python_path,
                                                               'jobs_path': self.jobs_path,
                                                               'conf_code': DagBuilder.CONF_CODE,
                                                               '3': "train_skip",
                                                               '5': "train"},
                                                    provide_context=True,
                                                    dag=globals()[dag_id])

            task_train_skip = DummyOperator(task_id="train_skip", dag=globals()[dag_id])

            task_train = self.create_template_task(dag_id, account_id, 'train')

            task_evaluate = self.create_template_task(dag_id, account_id, 'evaluate')

            task_predict = self.create_template_task(dag_id, account_id, 'predict', 'none_failed')


            # DAG

            task_start >> task_check_model >> [ task_train_skip, task_train ]
            task_train >> task_evaluate >> task_predict
            task_train_skip             >> task_predict


dag_builder = DagBuilder()
dag_builder.create_account_dags()
