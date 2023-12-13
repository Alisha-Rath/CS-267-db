import re
import string
import sys
import math

indexFilee = "corpus.txt"

def gamma_decode(encoded):
    """Decode an Elias gamma encoded string into an integer."""
    
    num_zeroes = 0
    
    for char in encoded:
        if char == '0':
            num_zeroes += 1
        else:
            break

    binary_representation = encoded[num_zeroes:num_zeroes * 2 + 1]
    
    return int(binary_representation, 2)

def parseFolderAndGet(file: str):
    fn = file
    try:
        with open(file, "r") as f:
            content = f.read().rstrip("\n")

        # with open('corpus.txt', "w") as f:
        #     f.write("<" + fn[::-1] + ">")

        return content
    
        

    except:
        print("Error: Unable to read file", file)
        return -1


def preProcessDocuments(data: str):
    translator = str.maketrans("", "", string.punctuation)
    data_without_punctuation = data.translate(translator)

    documents = data_without_punctuation.lower().split("\n\n")

    return documents

def string_to_binary(input_string):
    binary_string = ''.join(format(ord(char), '08b') for char in input_string)
    return binary_string

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
    return temp_inverted_index

def gamma_encode(number):
    """Encode a positive integer using Elias gamma coding."""
    if number < 0:
        raise ValueError("Elias gamma coding is only applicable for positive integers")
    
    binary_representation = bin(number)[2:]
    
    num_zeroes = len(binary_representation) - 1
    
    return '0' * num_zeroes + binary_representation


def gamma_decode(encoded):
    """Decode an Elias gamma encoded string into an integer."""
    
    num_zeroes = 0
    
    for char in encoded:
        if char == '0':
            num_zeroes += 1
        else:
            break

    binary_representation = encoded[num_zeroes:num_zeroes * 2 + 1]
    
    return int(binary_representation, 2)


def vbyte_encode(number):
    """Encode an integer using vByte encoding and return a string."""
    encoded_bytes = []
    while True:
        byte = number % 128
        if number < 128:
            encoded_bytes.insert(0, byte + 128)  # Mark the last byte
            break
        else:
            encoded_bytes.insert(0, byte)
        number //= 128

    # Convert each byte to its binary string representation and concatenate
    return ''.join(format(byte, '08b') for byte in encoded_bytes)


def represent_as_m_bit_binary(number, m):
    if number < 0:
        raise ValueError("Number must be non-negative")

    binary_representation = bin(number)[2:]  # Convert to binary and strip the '0b' prefix

    if len(binary_representation) > m:
        raise ValueError("Number requires more than m bits for representation")

    # Pad with zeros if necessary to ensure the string is m characters long
    return binary_representation.zfill(m)

def generate_dic_file(docs, index_filename):
    """Generate a.dic file for the inverted index."""

    global maxDocuments

    m = calculate_bit_length(docs)

    with open(index_filename + ".dic", "w") as f:
        m = calculate_bit_length(docs)
        f.write(gamma_encode(maxDocuments))
        f.write(gamma_encode(m))
        for doc in docs:
            value = len(doc.split(" "))
            binary_rep = represent_as_m_bit_binary(value, m)
            f.write(binary_rep)
            # f.write(" ")

def calculate_bit_length(documents):
    max_term_count = max(len(doc.split()) for doc in documents)
    return math.floor(math.log2(max_term_count)) + 1

def represent_as_m_bit_binary(number, m):
    if number < 0:
        raise ValueError("Number must be non-negative")

    binary_representation = bin(number)[2:]  # Convert to binary and strip the '0b' prefix

    if len(binary_representation) > m:
        raise ValueError("Number requires more than m bits for representation")

    # Pad with zeros if necessary to ensure the string is m characters long
    return binary_representation.zfill(m)

def append_unary(number, input_str, start_bit_offset, just_bit_offset=False):
    total_bits = start_bit_offset + number
    if just_bit_offset:
        start_bit_offset = total_bits
        return input_str

    bits_last_byte = total_bits & 7
    total_chars = total_bits >> 3 if bits_last_byte == 0 else (total_bits >> 3) + 1
    bits_first_byte = start_bit_offset & 7
    start_char = start_bit_offset >> 3

    output = input_str.ljust(total_chars, '\x00')
    start_char_ord = ord(output[start_char]) if start_char < len(output) else 0
    start_char_ord &= ((1 << bits_first_byte) - 1) << (8 - bits_first_byte)
    output = output[:start_char] + chr(start_char_ord) + output[start_char + 1:]

    last_ord = ord(output[total_chars - 1])
    last_ord += 1 if bits_last_byte == 0 else (1 << (8 - bits_last_byte))
    output = output[:total_chars - 1] + chr(last_ord) + output[total_chars:]

    start_bit_offset = total_bits
    return output

import math

def append_bits(number, input_str, start_bit_offset, num_bits=-1):
    start_char = start_bit_offset >> 3
    num_bits = math.ceil(math.log(number + 1, 2)) if num_bits == -1 else num_bits
    total_bits = start_bit_offset + num_bits
    bits_last_byte = total_bits & 7
    total_chars = total_bits >> 3 if bits_last_byte == 0 else (total_bits >> 3) + 1
    number &= (1 << num_bits) - 1

    output = input_str.ljust(total_chars, '\x00')
    cur_char = total_chars - 1
    cur_bits = number & ((1 << bits_last_byte) - 1)
    number >>= bits_last_byte
    start_remaining_bits = num_bits - bits_last_byte
    shift_last_byte = 0 if bits_last_byte == 0 else 8 - bits_last_byte
    output = output[:cur_char] + chr(ord(output[cur_char]) + (cur_bits << shift_last_byte)) + output[cur_char + 1:]
    cur_char -= 1

    for remaining_bits in range(start_remaining_bits, 7, -8):
        output = output[:cur_char] + chr(number & 255) + output[cur_char + 1:]
        cur_char -= 1
        number >>= 8

    if start_remaining_bits > 0:
        start_char_ord = ord(output[start_char])
        start_char_ord &= 255 - (1 << (start_remaining_bits - 1))
        output = output[:start_char] + chr(start_char_ord + number) + output[start_char + 1:]

    start_bit_offset = total_bits
    return output

def append_rice_sequence(int_sequence, modulus, output, start_bit_offset, delta_start=-1):
    last_encode = delta_start
    output, start_bit_offset = append_unary(modulus, output, start_bit_offset)
    mask = (1 << modulus) - 1
    
    for pre_to_encode in int_sequence:
        to_encode = pre_to_encode if delta_start < 0 else pre_to_encode - last_encode
        to_encode -= 1
        last_encode = pre_to_encode
        output, start_bit_offset = append_unary((to_encode >> modulus) + 1, output, start_bit_offset)
        output, start_bit_offset = append_bits(to_encode & mask, output, start_bit_offset, modulus)
    
    return output

# Helper functions append_unary and append_bits need to be defined


def write_index_to_file(inverted_index, index_filename):
    global indexFilee
    txt_filename = f'{index_filename}.txt'
    with open(txt_filename, 'w') as txt_file:  # Open in text mode
        # Placeholder for offset 1 and offset 2, will overwrite later
        txt_file.write('0' * 8)  # Placeholder for offset 1 (8 chars)
        txt_file.write('0' * 8)  # Placeholder for offset 2 (8 chars)

        # Term data section (offsets and terms)
        term_data_section = ''
        posting_list_section = ''
        term_offsets = []

        for term, postings in inverted_index.items():
            prev_doc_pos= 0
            term_offsets.append(len(term_data_section))
            num_docs_with_term = len(postings)
            term_data = term + vbyte_encode(num_docs_with_term) + vbyte_encode(len(posting_list_section))
            term_data_section += term_data

            # Posting list data
            for doc_id, positions in postings.items():
                gap= doc_id - prev_doc_pos
                frequency = len(positions)  # Example, replace with actual frequency
                posting_list_section += gamma_encode(doc_id) + gamma_encode(frequency) + gamma_encode(gap)
                prev_doc_pos = doc_id

        # Calculate w
        max_offset = max(term_offsets)
        w = math.floor(math.log2(max_offset)) + 1 if max_offset > 0 else 1
        encoded_w = gamma_encode(w)

        # Write actual offsets for term data and posting list section
        offset_1 = len(encoded_w) + len(term_offsets) * w
        offset_2 = len(term_data_section)
        txt_file.seek(0)
        txt_file.write(gamma_encode(offset_1))
        txt_file.write(gamma_encode(offset_2))

        # Write w and term offsets
        txt_file.seek(16)  # Move past the placeholders (each gamma_encode is 8 chars)
        txt_file.write(encoded_w)
        for offset in term_offsets:
            txt_file.write(format(offset, f'0{w}b'))  # Encode each offset with w bits

        # Write term data and posting list data
        txt_file.write(term_data_section)
        txt_file.write(posting_list_section)
        txt_file.write(f"/0r{string_to_binary(indexFilee)}")

def main():
    global inverted_index, maxDocuments, indexFilee

    if len(sys.argv) < 3:
        print("Invalid Usage")
        print(
            "Example Usage: python index.py corpus_file.txt index_filename.txt"
        )
        return
    indexFilee= sys.argv[1]

    file = "./" + sys.argv[1]
    index_filename = sys.argv[2]

    content = parseFolderAndGet(file)
    documents = preProcessDocuments(content)

    inverted_index = generateInvertedIndex(documents)
    maxDocuments = len(documents)

    generate_dic_file(documents, index_filename)
    write_index_to_file(inverted_index, index_filename)

    # print(inverted_index)
    # print(index_filename)
    # print(gamma_encode(12), gamma_decode(gamma_encode(12)))

if __name__ == "__main__":
    main()
