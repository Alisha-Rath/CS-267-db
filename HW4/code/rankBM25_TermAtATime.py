import string
import sys
import re
import math
import numpy as np
import heapq
import timeit


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

    def isInf(self, other):
        if self.doc_id == float("inf") or self.pos == float("inf"):
            return True
        return False

    def isNegInf(self, other):
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

                    if document_number in temp_inverted_index[term]:
                        temp_inverted_index[term][document_number].append(term_number)
                    else:
                        temp_inverted_index[term][document_number] = [term_number]
            self.document_length[document_number] = len(terms)
            self.total_corpus_term += len(terms)

        self.document_average_length = self.total_corpus_term / self.number_of_document
        return temp_inverted_index

    def is_present(self, term) -> bool:
        if term in self.inverted_index:
            return True
        return False

    def binary_search_next(self, P, low, high, pos) -> float:
        """Perform a binary search to find the next position of a term in a document after a given position."""
        while low <= high:
            mid = (low + high) // 2
            if P[mid] == pos:
                return mid + 1 if mid + 1 < len(P) else float("inf")
            elif P[mid] < pos:
                low = mid + 1
            else:
                high = mid - 1
        return low if low < len(P) else float("inf")

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

    def next(self, term, current) -> Position:
        docid, pos = current.doc_id, current.pos
        # P = posting list for terms in docid

        P = self.inverted_index.get(term, {}).get(docid, [])
        l = len(P)

        if l == 0 or P[-1] <= pos:
            return Position(docid, float("inf"))
        if P[0] > pos:
            return Position(docid, P[0])

        if l > 1 and P[l - 2] <= pos:
            low = l - 2
        else:
            low = 0

        jump = 1
        high = low + jump
        while high < l and P[high] <= pos:
            low = high
            jump *= 2
            high = low + jump

        if high >= l:
            high = l - 1

        c = self.binary_search_next(P, low, high, pos)
        position = P[c] if c != float("inf") else c

        return Position(docid, position)

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
            if next_pos.doc_id != float("inf"):
                return None
            else:
                # Keep finding the next term until the doc_id of the prev term is greater than current doc_id
                while (
                    next_pos.doc_id != float("inf")
                    and next_pos.doc_id <= current.doc_id
                ):
                    next_pos = self.next(term, next_pos)
                if not next_pos.isInf():
                    return next_pos.doc_id
        return None

    def get_document_frequency(self, term, docid):
        if self.is_present(term):
            return len(self.inverted_index[term][docid])
        return 0  # return 0 if the term is not in the vocabulary

    def get_term_doc_freq(self, terms):
        term_doc_freq = []
        term_document_list = {}
        for term in terms:
            posting_list = self.inverted_index[term]
            doc_set = set(posting_list.keys())

            doc_count = len(doc_set)
            term_doc_freq.append((doc_count, term))

            doc_list = sorted(list(doc_set))
            term_document_list[term] = doc_list

        term_doc_freq.sort()
        return term_doc_freq, term_document_list

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

    def get_top_k_results_heaps(self, acc, k):
        if len(acc) < k:
            k = len(acc)

        heap = []
        for result in acc.values():
            if result["docid"] < math.inf:
                heapq.heappush(heap, (result["score"], result["docid"]))
        top_k_results = heapq.nlargest(k, heap)
        return top_k_results

    def rank_bm25_term_at_a_time(self, terms, k, acc_num):
        # sort terms based on doc freq
        term_doc_freq, term_doc_list = self.get_term_doc_freq(terms)
        terms = [x[1] for x in term_doc_freq]

        # acc used for prev round, acc_p used for next
        acc = {}
        acc_p = {}
        acc[0] = {"docid": math.inf, "score": math.inf}
        for term in terms:
            if term not in self.inverted_index:
                break
            quota_left = acc_num - len(acc)
            in_pos = 0
            out_pos = 0
            if len(term_doc_list[term]) <= quota_left:
                for d in term_doc_list[term]:
                    while acc[in_pos]["docid"] < d:
                        acc_p[out_pos] = acc[in_pos].copy()
                        out_pos += 1
                        in_pos += 1
                    acc_p[out_pos] = {
                        "docid": d,
                        "score": self.TF_BM25(term, d),
                    }
                    if acc[in_pos]["docid"] == d:
                        acc_p[out_pos]["score"] += acc[in_pos]["score"]
                        in_pos += 1
                    out_pos += 1

            elif quota_left == 0:
                for j in range(len(acc)):
                    acc[j]["score"] = acc[j]["score"] * self.TF_BM25(
                        term, acc[j]["docid"]
                    )

            # copy remaining acc to acc'
            while acc[in_pos]["docid"] < math.inf:
                acc_p[out_pos] = acc[in_pos].copy()
                out_pos += 1
                in_pos += 1

            # end-of-list marker
            acc_p[out_pos] = {"docid": math.inf, "score": math.inf}
            acc, acc_p = acc_p, acc

        top_k_results = self.get_top_k_results_heaps(acc, k)
        return top_k_results


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
    num_accum = int(sys.argv[3])
    input_queries = sys.argv[4]

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

    start = timeit.default_timer()
    result = index.rank_bm25_term_at_a_time(query_terms, num_of_results, num_accum)
    stop = timeit.default_timer()

    print("Total Time elapsed: ", stop - start)

    # query_id = random.randrange(10)
    i = 0
    # print("query_id iter docno rank sim run_id")
    for r in result:
        if r[0] != 0:
            i = i + 1
            print("1", "0", r[1], i, r[0], "BM_TermAtATime")


if __name__ == "__main__":
    main()
