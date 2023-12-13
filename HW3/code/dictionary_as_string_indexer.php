<?php

namespace cs267\hw3;

use seekquarry\yioop\library\Library;
use seekquarry\yioop\library\FetchUrl;
use seekquarry\yioop\library\CrawlConstants;
use seekquarry\yioop\library\processors\HtmlProcessor;
use seekquarry\yioop\library\PhraseParser;
use seekquarry\yioop\library\PackedTableTools;


require_once "vendor/autoload.php";
Library::init();

$index = [];

function readUrls($filename) {

    if (!file_exists($filename)) {
        die("Error: The file {$filename} does not exist.\n");
    }

    $urls = file($filename);

    $stripped_urls = [];
    foreach ($urls as $url) {
        $stripped_url = str_replace(array("\n", " "),"",$url);
        array_push($stripped_urls, $stripped_url);
    }

    return $stripped_urls;
}

function getPagesInfo($urls) {
    # collect all the pages
    $pages=[];
    foreach ($urls as $url) {
        array_push($pages, [CrawlConstants::URL => $url]);
    }

    # download a collection of web pages and then pretty print
    $pages_info = FetchUrl::getPages($pages);

    return $pages_info;
}

$filename = $argv[1];
$output_file = $argv[2];

$urls = readUrls($filename);
$size = count($urls);
$pages_info = getPagesInfo($urls);

$htmlprocessor = new HtmlProcessor(
    $max_description_len=20000, 
    $summarizer_option= CrawlConstants::CENTROID_WEIGHTED_SUMMARIZER
);

foreach ($urls as $idx => $url) {
    $page = $htmlprocessor->process($pages_info[$idx][CrawlConstants::PAGE], $url);
    $summary = PhraseParser::extractWordStringPageSummary($page);
    $summary = rtrim($summary);
    $summary_array = explode(" ", $summary);

    // $page = $pages_info[$idx];
    // $processed_data = $htmlprocessor->process($page[CrawlConstants::PAGE], $page[CrawlConstants::URL]);
    // $summary_array = explode(" ", $processed_data[CrawlConstants::DESCRIPTION]);

    for($i=0; $i<count($summary_array); $i++) {
        $summary_array[$i] = strtolower($summary_array[$i]);
    }
    $summary_array = preg_replace('/[^a-z]+/i', '', $summary_array);
    $summary_array = array_filter($summary_array);

    $summary = implode(" ", $summary_array);
   
    // $stemmed_terms = $summary_array;

    $stemmed_terms = PhraseParser::stemTerms($summary, $page[CrawlConstants::LANG]);
    

    foreach ($stemmed_terms as $ind=>$stemmed_term) {
        if (!isset($index[$stemmed_term])) {
            $index[$stemmed_term] = array();
        }
    
        if (!isset($index[$stemmed_term][$idx])) {
            $index[$stemmed_term][$idx] = array();
        }
    
        $index[$stemmed_term][$idx][] = $ind;
        
    }
}


// Prepare the data to be written to the file
$inverted_index_string = '';

foreach ($index as $term => $doc_list) {
    $inverted_index_string .= $term . ":{";
    foreach ($doc_list as $docID => $positions) {
        $inverted_index_string .= $docID . "=>[" . implode(',', $positions) . "],";
    }
    $inverted_index_string = rtrim($inverted_index_string, ",") . "} ";
}

$stemmed_terms_string = implode(" ", array_keys($index));


// Get the length of the number of terms and the length of the inverted index string
$length_of_number_of_terms = strlen((string)$stemmed_terms_string);
$length_of_inverted_index = strlen($inverted_index_string);

$format = [
    "PRIMARY KEY" => "DOC_ID",
    "length_of_number_of_terms" => "INT",
    "length_of_inverted_index" => "INT",
    "docs_count" => "INT",
];

// Prepare the data
$data_to_pack = [
    [
        "length_of_number_of_terms" => $length_of_number_of_terms,
        "length_of_inverted_index" => $length_of_inverted_index,
        "docs_count" => $size
    ]
];

// Initialize the PackedTableTools instance
$packed_tools = new PackedTableTools($format);

// Pack the data
$packed_data = $packed_tools->pack($data_to_pack);

// ... (previous code)

// Write the packed data to the file
$file_handle = fopen($output_file, "w");


fwrite($file_handle, $packed_data);

// Append the stemmed terms string to the file.
fwrite($file_handle, $stemmed_terms_string);

// Append the inverted index string to the file.
fwrite($file_handle, $inverted_index_string);

// fwrite($file_handle, PHP_EOL); 

fclose($file_handle);

echo "Data written to " . $output_file . "\n";
