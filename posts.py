

class Posts:
    """
    Class creating objects of relevant data from Posts
    """
    __Id = int
    __Body = str
    __Title = str
    __Score = int
    __ViewCount = int
    __ClosedDate = str
    __AcceptedAnswerId = int

    def __init__(self, post_id, body, title, score, view_count, closed_date, accepted_answer_id):
        self.__Id = post_id
        self.__Body = body
        self.__Title = title
        self.__Score = score
        self.__ViewCount = view_count
        self.__ClosedDate = closed_date
        self.__AcceptedAnswerId = accepted_answer_id

    def get_id(self):
        return self.__Id

    def get_body(self):
        return self.__Body

    def get_title(self):
        return self.__Title

    def get_score(self):
        return self.__Score

    def get_view_count(self):
        return self.__ViewCount

    def get_closed_date(self):
        return self.__ClosedDate

    def get_accepted_answer_id(self):
        return self.__AcceptedAnswerId
