# from ruamel.yaml import YAML
# yaml = YAML(typ='unsafe')
# yaml.register_class(HParams)
#
# with open('test.yaml', 'w') as fp:
#     yaml.dump(hparams, fp)

# from ruamel.yaml import YAML
#
# yaml = YAML()
#
#
#
# with open('settings/app.yaml') as fp:
#     r = yaml.load(fp)['debug']
#     h = HParams()
#     for k, v in r.items():
#         h.add_hparam(k, v)
#
#     print(h)
from utils.parameter import AppConfig

appconfig = AppConfig('settings/app.yaml', 'debug')
print(appconfig.values())
print(appconfig.data_dir)
print(appconfig.log_dir)
print(appconfig.model_parameter.optimizer)
