import openai
import os
import time

openai.api_key = ""
for txtp in [
    "record.txt",
]:
    save_line = ""
    with open(txtp, "r") as f:
        lines = f.readlines()
        for line in lines:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Now you are a doctor, please extract the keywords of the relevant symptoms that the patient suffers from, separate each keyword with a space, just output the keywords."
                    },
                    {"role": "user", "content": line},
                ],
            )
            message = response["choices"][0]["message"]["content"]
            save_line = (
                message.replace("、", " ")
                .replace("，", " ")
                .replace("。", "")
                .replace("症状关键词：", "")
                .replace(",", " ")
                .replace("关键词：", "")
            )
            save_line = "".join([save_line, "\n"])
            print(message)
            print(save_line)
            print(response["usage"])
            save_line = bytes((save_line), encoding="utf-8")
            with open(txtp.replace(".", "_save."), "ab") as fsave:
                fsave.write(save_line)
            time.sleep(1)
        pass

    pass
