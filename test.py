from component.llms import Openai_model

if __name__ == "__main__":
    model=Openai_model()
    message = "中华人民共和国消费者权益保护法什么时候,在哪个会议上通过的？"
    result=model.chat(message)
    print(result) 
