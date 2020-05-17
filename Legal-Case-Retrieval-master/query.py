class Query(object):
    def __init__(self, q_type, data):
        self.q_type = q_type
        self.data = data

    def print_query(self):
        print("type: %s \n" % self.q_type)
        print("data: %s \n" % self.data)