from enum import Enum
import collections
import heapq
class Message(object):

    def __init__(self, MessageType, value=None, iteration=None, certificate=None, sender=None, receiver='BROADCAST'):
        self.messageType = MessageType
        self.value = value
        self.iteration = iteration
        self.certificate = certificate
        self.sender = sender
        self.receiver = receiver
    
    def __str__(self):
        messageString = str(self.messageType) + ", " + str(self.value) + ", " + str(self.iteration) + ", " + str(self.certificate) + ", " + str(self.sender) + ", " + str(self.receiver)
        return messageString
    
    def __lt__(self, other):
        return self.iteration < other.iteration

    def __cmp__(self, other):
        return self.iteration < other.iteration

    def __repr__(self):
        return str(self)

class MessageType(Enum):
    STATUS = 1
    PROPOSE = 2
    VOTE = 3
    COMMIT = 4
    NOSEND = 5


if __name__ == "__main__":
    m1 = Message(MessageType.STATUS, "value-0", -5, "cert", "agent-0", "agent-1")
    m2 = Message(MessageType.STATUS, "value-0", 6,  "cert", "agent-0", "agent-1")
    # m3 = Message(MessageType.STATUS, "value-0", 25, "sig", "cert", "agent-0", "agent-1")
    # messages = []
    # heapq.heappush(messages, m1)
    # heapq.heappush(messages, m2)
    # heapq.heappush(messages, m3)
    # heapq.heappush(messages, m4)
#     m1 = Message()

#     print(m1)
    state = [1,2,3]
    log = dict()
    log["m"] = []
    log["m"].append((state, m1))
    print(log)
    # var = []
    # print(m1)
    # var.append(m1)
    # var.append(m2)
    # print(var)
    # actionStr = 'send_agent-1_v-2_agent-3_v-0'
    # receiver_id = 1
    # if 'agent-' + str(receiver_id) in actionStr:
    #     value = actionStr.split('agent-'+str(receiver_id)+'_v-')[-1][0]
    #     print(value)
