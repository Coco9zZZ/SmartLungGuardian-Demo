import json


class Result:
    def __init__(self, status, data, msg):
        self.status = status
        self.data = data
        self.msg = msg

    def to_dict(self):
        return {
            "status": self.status,
            "data": self.data,
            "msg": self.msg
        }

    @staticmethod
    def success(data):
        return json.dumps(Result(200, data, '请求成功').to_dict(), ensure_ascii=False)

    @staticmethod
    def error(msg='请求失败'):
        return json.dumps(Result(400, '', msg).to_dict(), ensure_ascii=False)
