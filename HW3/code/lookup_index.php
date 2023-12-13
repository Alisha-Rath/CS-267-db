<?php

namespace cs267\hw3;

// require_once "./recontructInvertedIndex.php";

use seekquarry\yioop\library\CrawlConstants;
use seekquarry\yioop\library\Library;
use seekquarry\yioop\library\LinearAlgebra;
use seekquarry\yioop\library\PhraseParser;
use seekquarry\yioop\library\PackedTableTools;

// use cs267\hw3\readFile\reconstructInvertedIndex;

require_once "vendor/autoload.php";
Library::init();


function reconstructInvertedIndex($output_file) {
    if (!file_exists($output_file)) {
        die("Error: The file {$output_file} does not exist.\n");
    }

    $file_handle = fopen($output_file, "r");

    // Read packed data
    $packed_data = fread($file_handle, filesize($output_file));  // This reads the entire file

    // Define the format for unpacking
    $format = [
        "PRIMARY KEY" => "DOC_ID",
        "length_of_number_of_terms" => "INT",
        "length_of_inverted_index" => "INT",
        "docs_count" => "INT",

    ];

    // Initialize PackedTableTools with the format
    $packed_tools = new PackedTableTools($format);

    // Unpack the data
    $unpacked_data = $packed_tools->unpack($packed_data);

    // Retrieve the lengths
    $length_of_number_of_terms = $unpacked_data[0]["length_of_number_of_terms"];
    $length_of_inverted_index = $unpacked_data[0]["length_of_inverted_index"];
    $docs_count = $unpacked_data[0]["docs_count"];

    // Calculate positions to read stemmed_terms_string and inverted_index_string
    $total_data_length = strlen($packed_data);
    $position_for_stemmed_terms_string = $total_data_length - $length_of_number_of_terms - $length_of_inverted_index;
    $position_for_inverted_index_string = $total_data_length - $length_of_inverted_index;

    // Read the stemmed terms and inverted index strings
    fseek($file_handle, $position_for_stemmed_terms_string);  // Set pointer to the start of stemmed_terms_string
    $stemmed_terms_string = fread($file_handle, $length_of_number_of_terms);

    fseek($file_handle, $position_for_inverted_index_string);  // Set pointer to the start of inverted_index_string
    $inverted_index_string = fread($file_handle, $length_of_inverted_index);

    // Close the file
    fclose($file_handle);

    // Output the data
    // echo "Length of number of terms: " . $length_of_number_of_terms . "\n";
    // echo "Length of inverted index: " . $length_of_inverted_index . "\n";
    // echo "Stemmed terms string: " . $stemmed_terms_string . "\n";
    // echo "Inverted index string: " . $inverted_index_string . "\n";


    // Initialize the inverted index array
    $inverted_index = [];

    // Split the inverted index string by spaces to get each term's representation
    $terms_representations = explode(" ", $inverted_index_string);

    $terms_representations = array_filter($terms_representations);

    // print_r($terms_representations);

    foreach ($terms_representations as $term_representation) {
        // Split by colon to separate the term from its document list
        list($term, $doc_list_string) = explode(":", $term_representation);

        
        // Remove the opening and closing braces from the document list string
        $doc_list_string = trim($doc_list_string, "{}");
        
        // Split the document list string by comma to get individual docID to position mappings
        $doc_mappings = explode(",", $doc_list_string);
        
        foreach ($doc_mappings as $doc_mapping) {
            // Split the mapping by '=>' to separate the docID from the positions
            list($docID, $positions_string) = explode("=>", $doc_mapping);
            
            // Convert docID string to integer
            $docID = intval($docID);
            
            // Remove the opening and closing brackets from the positions string and split by comma
            $positions = explode(",", trim($positions_string, "[]"));
            
            // Convert position strings to integers
            $positions = array_map('intval', $positions);
            
            // Store the positions in the inverted index
            $inverted_index[$term][$docID] = $positions;
        }
    }


    return [$stemmed_terms_string, $inverted_index , $docs_count];
}


$fileName = $argv[1];
$searchQuery = $argv[2];

$fileData = reconstructInvertedIndex($fileName);
$terms_array= explode(" ",$fileData[0]);
$index = $fileData[1];

sort($terms_array);

// print_r($terms_array[0]);
// print_r($index);

$search_terms = PhraseParser::stemTerms($searchQuery, 'en-US');
// $search_terms = explode(" ", $searchQuery);

// docs_tf stores the document vectors [tf+1 for each term in corpus], ...]
$docs_tf = [];
$doc_count = $fileData[2];
$term_count = count($terms_array);

 for($i=0; $i<count($search_terms); $i++) {
        $search_terms[$i] = strtolower($search_terms[$i]);
}


$search_terms = preg_replace('/[^a-z]+/i', '', $search_terms);
$search_terms = array_filter($search_terms);


for ($i = 0; $i < $doc_count; $i++) {
    for ($j = 0; $j < $term_count; $j++) {
        $docs[$i][$j] = 0;
    }
}

foreach($index as $term=>$val) {
	foreach($val as $docid=>$pos) {
        $term_index = array_search($term, $terms_array);
        $docs_tf[$docid][$term_index] = log(count($pos), 2) + 1;
    }
}

$idf = [];

foreach($terms_array as $ind=>$term) {
    $idf[$ind] = log($doc_count/count($index[$term]), 2);
}


// print_r($idf);


$docs = [];
for ($i = 0; $i < $doc_count; $i++) {
    $docs[$i] = LinearAlgebra::multiply($docs_tf[$i], $idf);
}

// create query vector

$query_vector = [];

for($i=0;$i<count($terms_array);$i++) {
    $query_vector[$i] = 0;
}

$termCount = array_count_values($search_terms);

foreach ($search_terms as $ind=>$term) {
    $term_index = array_search($term, $terms_array);   
    $query_vector[$term_index] = log($termCount[$term],2)+1;
}

// print_r($query_vector);

$query_vector = LinearAlgebra::multiply($query_vector, $idf);

$ranks = array();


for($i=0;$i<$doc_count;$i++) {
    $score = LinearAlgebra::similarity($docs[$i], $query_vector);
    $ranks[$i] = array("docId"=>$i, "score"=>$score);
}

$col = array_column( $ranks, "score" );
array_multisort($col,SORT_ASC,$ranks );
$ranks = array_reverse($ranks);

// print_r($ranks);

foreach ($ranks as $idx => $docScore) {
    echo "(" . $docScore['docId'] . " , " . $docScore['score'] . " )\n";
}