import importlib
from models.base_model import BaseModel

def find_model_using_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported. 
    model_filename = "models." + model_name + "_model"  # models.pix2pix_model  相当于指明导入的包的具体位置
    modellib = importlib.import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '') + 'model'   # CDmodel
    for name, cls in modellib.__dict__.items():    # 用于遍历 modellib 模块的所有属性和类
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):          #   lower()这个条件检查 name 变量(通常是一个字符串)转换为小写后是否等于 target_model_name 转换为小写后的值   \实现换行
            model = cls
            
    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options

def create_model(opt):
    model = find_model_using_name(opt.model) # opt.model =  pix2pix 查找模型的文件的.py文件 并且返回模型的整个类
    instance = model()
    instance.initialize(opt)
    print("model [%s] was created" % (instance.name()))
    return instance
    
    
