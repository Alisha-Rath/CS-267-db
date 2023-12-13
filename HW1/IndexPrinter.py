import string
import sys
import re

inverted_index = {}


def main():
    if len(sys.argv) < 2:
        print("Usage: python IndexPrinter.py filename.txt")
        return
    
    file = sys.argv[1]

    with open(file, 'r') as f:
        content = f.read().rstrip("\n").lower()

    documents = content.split("\n\n")


    inverted_index = generateInvertedIndex(documents)
    print(inverted_index)

    # format_and_print(inverted_index)



def generateInvertedIndex(documents):
    temp_inverted_index = {}
    for document_number, document in enumerate(documents):
        document = re.sub(r'(?<=[a-zA-Z])[\W\d_]+(?=[a-zA-Z])|\n', ' ', document).strip()
        terms = document.split(" ")
        
        for term_number, term in enumerate(terms):
            if not any(char in string.punctuation for char in term):
                if  term not in temp_inverted_index:
                    temp_inverted_index[term]={}
                
                if  document_number in temp_inverted_index[term]:
                    temp_inverted_index[term][document_number].append(term_number)
                else:
                    temp_inverted_index[term][document_number] = [term_number]

    print(temp_inverted_index)
    return temp_inverted_index



def binary_search(t, low, high, current, post_list, next = True):
    post_list_for_term = post_list[t]
    position = current[1]
    doc = current[0]
    #mini = min(i for i in post_list_for_term if i >= current)
    while high - low > 1:
        mid = (low+high)//2
        if post_list_for_term[mid][0] <= doc and post_list_for_term[mid][1] <= position:
            low = mid
        else:
            high = mid
    #return post_list_for_term.index(mini)
    if next:
        return high
    else:
        return low


def format_and_print(inverted_index):
    sorted_index = {key: values for key, values in sorted(inverted_index.items())}

    total_unique_words = len(sorted_index)

    first_string=f"{total_unique_words:04}"
    second_string=""
    third_string=""

    for term in sorted_index:
        third_index = len(third_string)
        second_index= len(second_string)

        temp_third = ','.join(str(v) for v in sorted_index[term])
       
        third_string +=temp_third +','

        second_string += term+f"{third_index:04}"

        first_string += f"{second_index:04}"

    third_string = third_string[:-1]

    processed_output = first_string+'\n'+second_string+'\n'+third_string 
    print(processed_output)
    print(inverted_index)

if __name__ == "__main__":
    main()
