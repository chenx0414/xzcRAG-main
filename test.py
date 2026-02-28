from component.llms import Openai_model

if __name__ == "__main__":
    model=Openai_model()
    message = "豹女的技能是什么"
    result=model.chat(message)
    print(result) 
