import string
import sys
import re
import math
import heapq
import timeit
import operator


# This class holds the tuple (doc_id, position)
# Overloaded operator __gt__ (>), __lt__ (<), __ge__(>=), __le__(<=) __str__
# These operators allows to compare the position
class Position:
    def __init__(self, doc_id, position):
        self.doc_id = doc_id
        self.pos = position

    def __gt__(self, other):
        if self.doc_id > other.doc_id:
            return True
        elif self.doc_id == other.doc_id and self.pos > other.pos:
            return True
        return False

    def __lt__(self, other):
        if self.doc_id < other.doc_id:
            return True
        elif self.doc_id == other.doc_id and self.pos < other.pos:
            return True
        return False

    def __ge__(self, other):
        if self.doc_id > other.doc_id:
            return True
        elif self.doc_id == other.doc_id and (
            self.pos > other.pos or self.pos == other.pos
        ):
            return True
        return False

    def __le__(self, other):
        if self.doc_id < other.doc_id:
            return True
        elif self.doc_id == other.doc_id and (
            self.pos < other.pos or self.pos == other.pos
        ):
            return True
        return False

    def isInf(self):
        if self.doc_id == float("inf") or self.pos == float("inf"):
            return True
        return False

    def isNegInf(self):
        if self.doc_id == float("-inf") or self.pos == float("-inf"):
            return True
        return False

    def __str__(self):
        return str(self.doc_id) + "," + str(self.pos)


class InvertedIndex:
    term_frequency = {}  # holds frequency of every term of the corpus
    document_frequency = {}  # holds document frequency of every term
    document_length = (
        {}
    )  # holds length of every document (length = number of terms in the document)
    total_corpus_term = 0  # total number of terms in the entire corpus
    document_average_length = 0
    number_of_document = 0
    last_index_position = {}
    temp_inverted_index = {}

    def __init__(self, documents):
        self.documents = documents
        self.inverted_index = self.generateInvertedIndex(documents)
        self.maxDocuments = len(documents)
        self.positional_index = {}
        self.number_of_document = len(documents)

    def generateInvertedIndex(self, documents):
        temp_inverted_index = {}
        self.number_of_document = len(documents)
        for document_number, document in enumerate(documents):
            document_number += 1
            document = re.sub(
                r"(?<=[a-zA-Z])[\W\d_]+(?=[a-zA-Z])|\n", " ", document
            ).strip()
            terms = document.split(" ")

            for term_number, term in enumerate(terms):
                term_number += 1
                if not any(char in string.punctuation for char in term):
                    if term not in temp_inverted_index:
                        temp_inverted_index[term] = {}
                        self.term_frequency[term] = 1
                        self.document_frequency[term] = 1

                    else:
                        self.term_frequency[term] += 1

                        if document_number in temp_inverted_index[term]:
                            temp_inverted_index[term][document_number].append(
                                term_number
                            )
                        else:
                            temp_inverted_index[term][document_number] = [term_number]
                            self.document_frequency[term] += 1
            self.document_length[document_number] = len(terms)
            self.total_corpus_term += len(terms)

        self.document_average_length = self.total_corpus_term / self.number_of_document
        return temp_inverted_index

    def is_present(self, term) -> bool:
        if term in self.inverted_index:
            return True
        return False

    def binary_search(self, term, b_low, b_high, current, op):
        while b_high - b_low > 1:
            mid = b_low + math.floor((b_high - b_low) / 2)

            mid_pos = self.find_position_for_cursor(term, mid)

            if op(mid_pos, current):
                b_low = mid
            else:
                b_high = mid

        return b_low, b_high

    def binary_search_prev(self, P, low, high, pos) -> float:
        """Perform a binary search to find the previous position of a term in a document before a given position."""
        ans = -float("inf")
        while low <= high:
            mid = (low + high) // 2
            if P[mid] < pos:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans if ans != -float("inf") else float("inf")

    def find_position_for_cursor(self, term, cursor):
        posting_list = self.inverted_index[term]

        for posting in posting_list:
            position_list = posting_list[posting]

            if cursor < len(position_list):
                # return when the cursor is appropriate index of the posting list
                return Position(posting, position_list[cursor])
            else:
                cursor -= len(position_list)
        return Position(float("inf"), float("inf"))

    def next(self, term, current: Position) -> Position:
        docid, pos = current.doc_id, current.pos
        # P = posting list for terms in docid

        firstPosition, lastPosition = self.first(term), self.last(term)

        P = self.inverted_index.get(term, {}).get(docid, [])
        l = len(P)

        if lastPosition is None or lastPosition <= current:
            return Position(docid, float("inf"))
        elif firstPosition > current:
            self.last_index_position[term] = 0
            return firstPosition

        low = 0

        if term in self.last_index_position and self.last_index_position[term] > 0:
            last_static_position = self.find_position_for_cursor(
                term, self.last_index_position[term] - 1
            )
            if last_static_position is not None and last_static_position <= current:
                # If the last position is present we begin the search from there
                low = self.last_index_position[term] - 1

        jump = 1
        high = low + jump

        high_pos = self.find_position_for_cursor(term, high)

        while (
            high < int(self.term_frequency[term])
            and high_pos is not None
            and high_pos <= current
        ):
            low = high
            jump = 2 * jump
            high = low + jump
            high_pos = self.find_position_for_cursor(
                term, high
            )  # Find the relative position of high cursor

        if high >= int(self.term_frequency[term]):
            high = int(self.term_frequency[term]) - 1

        low, high = self.binary_search(term, low, high, current, operator.lt)
        self.last_index_position[term] = high
        return self.find_position_for_cursor(term, self.last_index_position[term])

    def prev(self, term, current: Position) -> Position:
        """Find the previous position of a term in a document before a given position."""
        # P = posting list for terms in docid
        docid, pos = current.doc_id, current.pos
        P = self.inverted_index.get(term, {}).get(docid, [])
        l = len(P)

        if l == 0 or P[0] >= pos:
            return Position(docid, float("-inf"))
        if P[-1] < pos:
            return Position(docid, P[-1])

        if l > 1 and P[l - 2] >= pos:
            high = l - 2
        else:
            high = l - 1

        jump = 1
        low = high - jump
        while low >= 0 and P[low] >= pos:
            high = low
            jump *= 2
            low = high - jump

        if low < 0:
            low = 0

        c = self.binary_search_prev(P, low, high, pos)
        position = P[c] if c != float("-inf") else c
        return Position(docid, position)

    def nextPhrase(self, terms, pos: Position) -> Position:
        v = pos

        # For each term, find the next occurrence after the current position.
        for term in terms:
            v = self.next(term, v)
            if v.isInf():
                return Position(float("inf"), float("inf"))

        u = v

        # For each term except the last one, find the previous occurrence before the current position.
        for i in range(len(terms) - 2, -1, -1):
            u = self.prev(terms[i], u)

        if v.pos - u.pos == len(terms) - 1:
            print("Found the phrase here ", u, v)
            return u

        return self.nextPhrase(terms, u)

    def prevPhrase(self, terms, pos: Position) -> Position:
        """
        Returns the previous occurrence of a phrase (sequence of terms) before a given position.
        The returned value is in the form of docid:pos.
        """
        v = pos

        # For each term, starting from the last, find the previous occurrence before the current position.
        for i in range(
            len(terms) - 1, -1, -1
        ):  # Note: We're starting from the second-last term based on the pseudocode
            v = self.prev(terms[i], v)
            if v.isInf():
                return Position(float("-inf"), float("-inf"))

        u = v

        # For each term except the first one, find the next occurrence after the current position.
        for i in range(1, len(terms)):
            u = self.next(terms[i], u)

        if (u.pos - v.pos) == len(terms) - 1:
            return v

        return self.prevPhrase(terms, v)

    def get_doc_id(self, u: Position):
        return u.pos

    def docRight(self, termsList, currentDoc) -> float:
        for i in range(currentDoc, self.maxDocuments + 1):
            found = True

            u = Position(float("inf"), float("inf"))

            if not self.positional_index.get(i, False):
                self.positional_index[i] = {}

            for term in termsList:
                key = "_".join(term)
                u = self.nextPhrase(
                    term,
                    Position(
                        i, self.positional_index.get(i, {}).get(key, float("-inf"))
                    ),
                )
                if u.isInf():
                    found = False
                    break

                self.positional_index[u.doc_id][key] = u.pos

            if found:
                doc_id = self.get_doc_id(u)
                return doc_id

        return float("inf")

    def docLeft(self, termsList, currentDoc) -> float:
        for i in range(currentDoc, -1, -1):
            found = True

            u = Position(float("-inf"), float("-inf"))

            for term in termsList:
                u = self.prevPhrase(term, Position(i, float("inf")))
                if u.isNegInf():
                    found = False
                    break

            if found:
                doc_id = self.get_doc_id(u)
                return doc_id

        return float("-inf")

    def first(self, term):
        if self.is_present(term):
            return Position(
                self.first_doc(term),
                self.inverted_index[term][list(self.inverted_index[term].keys())[0]][0],
            )
        return None

    def last(self, term):
        if self.is_present(term):
            return Position(
                self.last_doc(term),
                self.inverted_index[term][list(self.inverted_index[term].keys())[0]][0],
            )
        return None

    def first_doc(self, term):
        if self.is_present(term):
            return list(self.inverted_index[term].keys())[0]
        return None

    def last_doc(self, term):
        if self.is_present(term):
            return list(self.inverted_index[term].keys())[-1]
        return None

    def next_doc(self, term, current: Position):
        if self.is_present(term):
            next_pos = self.next(term, current)
            if next_pos.isInf():
                return None
            else:
                # Keep finding the next term until the doc_id of the prev term is greater than current doc_id
                while not next_pos.isInf() and next_pos.doc_id <= current.doc_id:
                    next_pos = self.next(term, next_pos)
                if not next_pos.isInf():
                    return next_pos.doc_id
        return None

    def get_documents_by_term(self, term):
        documents = []
        if self.is_present(term):
            for posting_list in self.inverted_index[term]:
                documents.append(posting_list)
        return documents

    def get_document_frequency(self, term, docid):
        if self.is_present(term):
            return len(self.inverted_index[term][docid])
        return 0  # return 0 if the term is not in the vocabulary

    def TF_BM25(self, term, docid):
        k1 = 1.2
        b = 0.75
        ftd = self.get_document_frequency(term, docid)

        numerator = ftd * (k1 + 1)
        denominator = ftd + (
            k1
            * (
                (1 - b)
                + (b * (self.document_length[docid] / self.document_average_length))
            )
        )

        if denominator != 0:
            return float(numerator / denominator)
        return 0

    def IDF_BM25(self, term):
        Nt = self.get_documents_by_term(term)
        if len(Nt) == 0:
            return 0
        else:
            return math.log(float(self.number_of_document) / float(len(Nt)), 2)

    def calculateMaxScoreForTerms(self, query_terms):
        max_term_scores = []
        for t in query_terms:
            score = 2.2 * self.IDF_BM25(t)
            max_term_scores.append((t, score))
        return max_term_scores

    def find_BM25_rank_with_heap(self, query_terms, max_scores, k):
        results = []  # list for the top k document scores (score, doc_id)
        terms = []  # term heap contains tuple (doc_id, term)
        excluded_terms = set()
        for i in range(0, k):
            heapq.heappush(results, (0, 0.0))  # Making min-heap of size k

        # Creating term heap by increasing order of next_doc for the term
        for term in query_terms:
            heapq.heappush(terms, (self.first_doc(term), term))

        heapq.heapify(terms)

        t = heapq.nsmallest(1, terms)
        t = t[0]

        # Here self.number_of_documents+1 serves the purpose of +infinity
        while (
            t[0] != self.number_of_document + 1
        ):  # loop terminates when all the terms from heap are processed
            d = t[0]
            score = 0
            while t[0] == d:
                score = score + self.TF_BM25(t[1], d) * self.IDF_BM25(t[1])
                next_doc = self.next_doc(t[1], Position(d, 0))
                # When there does not exist next_doc for the term, we add number_of_document+1 to indicate +infinity
                if next_doc is None:
                    heapq.heapreplace(terms, (self.number_of_document + 1, t[1]))
                else:
                    heapq.heapreplace(terms, (next_doc, t[1]))
                heapq.heapify(terms)

                t = heapq.nsmallest(1, terms)
                t = t[0]

            # Process the excluded terms, because the scores are valid for documents having non-excluded terms
            for excluded_term in excluded_terms:
                doc = self.next_doc(excluded_term, Position(d - 1, 0))
                if doc is not None and doc == d:
                    score = score + self.TF_BM25(excluded_term, d) * self.IDF_BM25(
                        excluded_term
                    )

            # Get the smallest score from the heap, so that it can be compared with the max_scores list
            smallest = heapq.nsmallest(1, results)
            smallest_score = smallest[0][0]

            # We only add the score if it greater than the lowest score from the heap
            if score > smallest[0][0]:
                heapq.heapreplace(results, (score, d))
                heapq.heapify(results)

            # This loop checks if the smallest score from heap exceeds the max_score for query term
            # max_scores is list containing (max_score, term) for each query term in increasing order of max_score
            for max_score in max_scores:
                if smallest_score > max_score[1]:
                    terms = [
                        t for t in terms if t[1] != max_score[0]
                    ]  # remove term from the term heap
                    heapq.heapify(terms)
                    excluded_terms.add(max_score[0])  # add term to excluded terms
                    smallest_score = smallest_score - max_score[1]

            t = heapq.nsmallest(1, terms)
            t = t[0]

        results.sort(reverse=True)
        return results


class DocumentProcessor:
    def __init__(self):
        pass

    def parseFolderAndGet(self, file: str):
        try:
            with open(file, "r") as f:
                content = f.read().rstrip("\n")

            return content

        except:
            print("Error: Unable to read file", file)
            return -1

    def preProcessDocuments(self, data: str):
        translator = str.maketrans("", "", string.punctuation)
        data_without_punctuation = data.translate(translator)

        documents = data_without_punctuation.lower().split("\n\n")

        return documents

    def preProcessQueries(self, input_queries: list[str]):
        query_list = input_queries.split(" ")
        query_list[:] = (
            value for value in query_list if value != "_AND" and value != "_OR"
        )

        query_list = [x.lower() for x in query_list]
        query_terms = list(set(query_list))

        return query_terms


def main():
    file_path = "./" + sys.argv[1]
    num_of_results = int(sys.argv[2])
    input_queries = sys.argv[3]

    processor = DocumentProcessor()

    content = processor.parseFolderAndGet(file_path)
    documents = processor.preProcessDocuments(content)
    query_terms = processor.preProcessQueries(input_queries)

    index = InvertedIndex(documents)

    print("File", file_path)
    print("Number of results", num_of_results)
    print("Input queries", query_terms)

    for term in query_terms:
        if not index.is_present(term):
            query_terms.remove(term)

    max_scores = index.calculateMaxScoreForTerms(query_terms)

    max_scores.sort(key=lambda tup: tup[1])

    start = timeit.default_timer()
    result = index.find_BM25_rank_with_heap(query_terms, max_scores, num_of_results)
    stop = timeit.default_timer()

    print("Time Elapsed: ", stop - start)

    # query_id = random.randrange(10)
    i = 0
    # print("query_id iter docno rank sim run_id")
    for r in result:
        if r[0] != 0:
            i = i + 1
            print("1", "0", r[1], i, r[0], "rankMB25 with heap")


if __name__ == "__main__":
    main()
