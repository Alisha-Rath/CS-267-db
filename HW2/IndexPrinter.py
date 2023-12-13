import string
import sys
import re
import math
import numpy as np

inverted_index = {}
maxDocuments = 0
positional_index = {}


def parseFolderAndGet(file: str):
    try:
        with open(file, "r") as f:
            content = f.read().rstrip("\n")

        return content

    except:
        print("Error: Unable to read file", file)
        return -1


def preProcessDocuments(data: str):
    translator = str.maketrans("", "", string.punctuation)
    data_without_punctuation = data.translate(translator)

    documents = data_without_punctuation.lower().split("\n\n")

    return documents


def preProcessQueries(input_queries: list[str]):
    queries = []

    for query in input_queries:
        queries.append(query.split("_"))

    return queries


def main():
    global inverted_index, maxDocuments

    if len(sys.argv) < 4:
        print("Invalid Usage")
        print(
            "Example Usage: python IndexPrinter.py filename.txt num_of_results queries"
        )
        return

    file = "./" + sys.argv[1]
    num_of_results = int(sys.argv[2])
    input_queries = sys.argv[3:]

    queries = preProcessQueries(input_queries)

    print("File", file)
    print("Number of results", num_of_results)
    print("Input queries", queries)

    content = parseFolderAndGet(file)
    documents = preProcessDocuments(content)

    inverted_index = generateInvertedIndex(documents)
    maxDocuments = len(documents)

    print(rankCosine(queries, num_of_results))

    # print(nextPhrase(['this', "is"],
    #                  [0, 0]))  # Expected output: [0, 1] because "this is" starts from position 1 in docid 0
    # print(nextPhrase(['second', "document"], [1, -1]))  # Expected output: [float('inf'), float('inf')] because there's no "hello this" phrase
    #
    # print(prevPhrase(['this', 'first'], [0, 4]))  # Expected output: [0, 1]
    # print(prevPhrase(['hello', 'this'], [0, 2]))
    # print(nextPhrase(['second', "document"], [1, -1]))

    # print(docRight([['this', 'is'], ['hello']], 0))
    # print(docRight([["Halifax", "document"], ["this"]], 0))
    # print(docLeft([["second", "document"], ["this"]], 3))


def generateInvertedIndex(documents):
    temp_inverted_index = {}
    for document_number, document in enumerate(documents):
        document = re.sub(
            r"(?<=[a-zA-Z])[\W\d_]+(?=[a-zA-Z])|\n", " ", document
        ).strip()
        terms = document.split(" ")

        for term_number, term in enumerate(terms):
            if not any(char in string.punctuation for char in term):
                if term not in temp_inverted_index:
                    temp_inverted_index[term] = {}

                if document_number in temp_inverted_index[term]:
                    temp_inverted_index[term][document_number].append(term_number)
                else:
                    temp_inverted_index[term][document_number] = [term_number]
    # print(temp_inverted_index)
    return temp_inverted_index


def binary_search(t, low, high, current, post_list, next=True):
    post_list_for_term = post_list[t]
    position = current[1]
    doc = current[0]
    # mini = min(i for i in post_list_for_term if i >= current)
    while high - low > 1:
        mid = (low + high) // 2
        if post_list_for_term[mid][0] <= doc and post_list_for_term[mid][1] <= position:
            low = mid
        else:
            high = mid
    # return post_list_for_term.index(mini)
    if next:
        return high
    else:
        return low


def binary_search_next(P, low, high, pos):
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


def next(term, docid, pos):
    # P = posting list for terms in docid

    P = inverted_index.get(term, {}).get(docid, [])
    l = len(P)

    if l == 0 or P[-1] <= pos:
        return [docid, float("inf")]
    if P[0] > pos:
        return [docid, P[0]]

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

    c = binary_search_next(P, low, high, pos)
    position = P[c] if c != float("inf") else c

    return [docid, position]


def binary_search_prev(P, low, high, pos):
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


def prev(term, docid, pos):
    global inverted_index
    """Find the previous position of a term in a document before a given position."""
    # P = posting list for terms in docid
    P = inverted_index.get(term, {}).get(docid, [])
    l = len(P)

    if l == 0 or P[0] >= pos:
        return [docid, float("-inf")]
    if P[-1] < pos:
        return [docid, P[-1]]

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

    c = binary_search_prev(P, low, high, pos)
    position = P[c] if c != float("-inf") else c
    return [docid, position]


def nextPhrase(terms, pos):
    global inverted_index
    v = pos

    # For each term, find the next occurrence after the current position.
    for term in terms:
        v = next(term, v[0], v[1])
        if v[1] == float("inf"):
            return [float("inf"), float("inf")]

    u = v

    # For each term except the last one, find the previous occurrence before the current position.
    for i in range(len(terms) - 2, -1, -1):
        u = prev(terms[i], u[0], u[1])

    if v[1] - u[1] == len(terms) - 1:
        print("Found the phrase here ", u, v)
        return u
    else:
        return nextPhrase(terms, u)


def prevPhrase(terms, pos):
    """
    Returns the previous occurrence of a phrase (sequence of terms) before a given position.
    The returned value is in the form of docid:pos.
    """
    v = pos

    # For each term, starting from the last, find the previous occurrence before the current position.
    for i in range(
        len(terms) - 1, -1, -1
    ):  # Note: We're starting from the second-last term based on the pseudocode
        v = prev(terms[i], v[0], v[1])
        if v[1] == float("-inf"):
            return [float("-inf"), float("-inf")]

    u = v

    # For each term except the first one, find the next occurrence after the current position.
    for i in range(1, len(terms)):
        u = next(terms[i], u[0], u[1])

    if (u[1] - v[1]) == len(terms) - 1:
        # print("Found phrase in previous at", terms, v,u)
        return v
    else:
        return prevPhrase(terms, v)


def get_doc_id(u):
    return u[0]


def docRight(termsList, currentDoc):
    global positional_index

    for i in range(currentDoc, maxDocuments + 1):
        found = True

        u = float("inf")

        if not positional_index.get(i, False):
            positional_index[i] = {}

        for term in termsList:
            key = "_".join(term)
            u = nextPhrase(
                term, [i, positional_index.get(i, {}).get(key, float("-inf"))]
            )
            # print(i, term, u)
            if u == [float("inf"), float("inf")]:
                found = False
                break

            positional_index[u[0]][key] = u[1]

        if found:
            doc_id = get_doc_id(u)
            return doc_id

    return float("inf")


def docLeft(termsList, currentDoc):
    for i in range(currentDoc, -1, -1):
        found = True

        u = float("-inf")

        for term in termsList:
            u = prevPhrase(term, [i, float("inf")])
            if u == [float("-inf"), float("-inf")]:
                found = False
                break

        if found:
            doc_id = get_doc_id(u)
            return doc_id

    return float("-inf")


def freqTermInQuery(term, terms):
    count = 0
    for t in terms:
        if term == t:
            count += 1
    return count


def tfidf(freqTerm, totDocWithTerms):
    global maxDocuments  # Assuming totDoc is a global variable

    # Calculate TF-IDF using the given formula
    tf = math.log(freqTerm) + 1  # Term Frequency (TF)
    idf = math.log(maxDocuments / totDocWithTerms)  # Inverse Document Frequency (IDF)

    return tf * idf


def createVectorQuery(terms):
    global inverted_index
    vectorQuery = []
    for term in terms:
        qt = freqTermInQuery(term, terms)
        Nt = len(inverted_index.get(term, []))  # Corrected Nt calculation

        score = tfidf(qt, Nt)
        vectorQuery.append(score)

    return vectorQuery


def createVectorDoc(terms, doc_id):
    global inverted_index
    vectorDoc = []
    for term in terms:
        p = inverted_index.get(term, {}).get(
            doc_id, []
        )  # Corrected accessing position list
        ft = len(p)  # Corrected ft calculation

        Nt = len(inverted_index.get(term, {}))  # Corrected Nt calculation

        score = tfidf(ft, Nt)
        vectorDoc.append(score)

    return vectorDoc


def multiply_unit_vectors(unit_vector1, unit_vector2):
    # Ensure both vectors have the same length
    if len(unit_vector1) != len(unit_vector2):
        raise ValueError("Input vectors must have the same length")

    # Perform element-wise multiplication
    result_vector = [u1 * u2 for u1, u2 in zip(unit_vector1, unit_vector2)]
    return result_vector


def calculate_unit_vector(vector):
    # Calculate the magnitude (Euclidean norm) of the vector
    magnitude = np.linalg.norm(vector)

    # Ensure the magnitude is not zero to avoid division by zero
    if magnitude == 0:
        raise ValueError("Cannot calculate the unit vector for a zero vector.")

    # Calculate the unit vector by dividing each element by the magnitude
    unit_vector = vector / magnitude

    return unit_vector


def sim(query, doc_id):
    global inverted_index
    # Flatten the query array of arrays into a single array of strings
    terms = [item for sublist in query for item in sublist]

    # Calculate the vector representation for the document and the query
    vector_doc = createVectorDoc(terms, doc_id)
    vector_query = createVectorQuery(terms)

    # Calculate the unit vectors for the document and the query
    unit_vector_doc = calculate_unit_vector(vector_doc)
    unit_vector_query = calculate_unit_vector(vector_query)

    # Calculate the similarity score using the dot product of the unit vectors
    similarity_score = np.dot(unit_vector_doc, unit_vector_query)

    return similarity_score


def rankCosine(queries, k):
    global inverted_index
    results = []

    # Initialize document ID and get the first document containing the terms
    doc_id = docRight(queries, 0)
    print(doc_id)

    # If no document contains the terms, skip this query
    if doc_id == float("inf"):
        return results

    # Calculate the similarity score for each document
    doc_scores = {}
    while doc_id < maxDocuments:
        score = sim(queries, doc_id)
        doc_scores[doc_id] = score
        doc_id = docRight(queries, doc_id)

    # Sort the documents by score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # Add the top k documents to the results
    results.extend(sorted_docs[:k])

    # Sort the final result by score
    results = sorted(results, key=lambda x: x[1], reverse=True)

    return results[:k]


if __name__ == "__main__":
    main()
