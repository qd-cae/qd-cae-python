#############################################################################
#                               QD Works                                    #
#                               2017                                        #
#############################################################################

# Description   : This is just an example to extract tables for LS Dyna Keywords from the LSDyna Manual (PDF)
#                 The JSON file in res folder consists of keys only for Manual 1. One may modify or run the
#                 same script to obtain the keywords page_numbers JSON file for Manual 2.

# Developed By  : N. Praba
# Date          : 21 Nov, 2017

import os
import tabula
import PyPDF2 as pdf
import json

REPLACE_CHARACTERS = ['\n', '\t']

def get_page_key(page_num, ls_file_contents):
	'''
	@description		: obtain the keyword from the page; it's considered to be a part of the header of the page and starting with *

	@ls_file_contents	: pypdf2 object that has contents of the pdf
	@returns			: returns a string consisting of the keyword name. If nothing was found, an empty string is returned.
	'''
	#get only the headers of the page - most likely the first two words in the page_contents list; also page nos. for this function start from 0
	page_contents = ls_file_contents.getPage(page_num-1).extractText().split(' ')[0:2]
	if not page_contents:
		return ""

	page_key = ""
	#isolate the correct key
	for text in page_contents:
		if "*" in text:
			#considering two cases - [*AIRBAG, *AIRBAG] and [*AIRBAG, *AIRBAG_SIMPLE_PRESSURE_VOLUME]
			if text != page_key and len(text) > len(page_key):
				for rchar in REPLACE_CHARACTERS:
					text = text.replace(rchar, '')

				page_key = text

	return page_key


def get_pdf_dict(lsdyna_manual_file_path, page_numbers, only_page_num_dict=False):
	'''
	@description				: returns a dictionary of keywords and corresponding tables and page numbers in lists

	@lsdyna_manual_file_path	: the path to the LS_Dyna Manual
	@page_numbers				: list of page numbers for which keys and tables must be extracted; if "All" is given, all pages are analyzed
	@only_page_num_dict			: (default:False) if this is set to True, only page numbers are written out for given key word manual
	@returns					: returns a dictionary of keywords and corresponding tables and page numbers in lists
	'''

	ls_key_dict = {}
	ls_file = open(lsdyna_manual_file_path, 'rb')
	ls_file_contents = pdf.PdfFileReader(ls_file)

	if isinstance(page_numbers, int):
		page_numbers = [page_numbers]

	if page_numbers == "All":
		total_pages = ls_file_contents.getNumPages()
		page_numbers = list(range(1, total_pages+1))

	print("\nLooking for tables in the pages - " + str(page_numbers) + ".\n")

	error_pages = []
	for page_num in page_numbers:
		page_key = get_page_key(page_num, ls_file_contents)
		if page_key == "":
			continue

		if ls_key_dict.get(page_key):
			ls_key_dict[page_key]['page_numbers'].append(page_num)
		else:
			ls_key_dict[page_key] = {'tables' : [], 'page_numbers' : [page_num]}

		if only_page_num_dict:
			continue

		#page nos. for this function start from 1
		try:
			page_tables = tabula.read_pdf(lsdyna_manual_file_path, pages=[page_num], multiple_tables=True)
		except:
			error_pages.append(page_num)
			page_tables = None
		if not page_tables:
			ls_key_dict[page_key]['page_numbers'].pop(-1)
			continue

		#collecting only LS Dyna key cards (the first cell in the table has the phrase "Card")
		valid_page_tables = []
		for table in page_tables:
			try:
				if "Card" in table[0][0]:
					#TODO: this area is meant to cleanup tables; try the keycard *DEFINE_COORDINATE_SYSTEM and notice table on Page 1560
					valid_page_tables.append(table)
			except:
				continue

		if valid_page_tables:
			ls_key_dict[page_key]['tables'].append(valid_page_tables)
		else:
			ls_key_dict[page_key]['page_numbers'].pop(-1)

	if error_pages:
		print("There were errors while extracting tables from pages " + str(error_pages))

	return ls_key_dict


def get_tables_lskeyword(ls_keyword, pages_numbers_only=False):
	'''
	@description		: returns the appropriate tables for the given keyword

	@ls_keyword			: ls dyna keyword (eg. *MAT_SPOTWELD_DAIMLERCHRYSLER)
	@page_numbers_only	: (default:False) returns only the pages belonging to the given keyword irrespective of tables present or not
	@returns			: a dictionary consisting of two lists - {'page_numbers' : [], 'tables' : []}
	'''
	lsdyna_manual_file_path = "LSDyna_Manual_1_2017.pdf"
	keyword_dict_path = lsdyna_manual_file_path.replace(".pdf", ".json")

	ls_key_dict = {}
	if not os.path.isfile(keyword_dict_path):
		ls_key_dict = get_pdf_dict(lsdyna_manual_file_path, "All", True)

		with open(keyword_dict_path, 'w') as outfile:
			json.dump(ls_key_dict, outfile)
		outfile.close()
	else:
		json_data = open(keyword_dict_path, 'r')
		ls_key_dict = json.load(json_data)
		json_data.close()

	if not ls_key_dict:
		print("No keywords were found.")

	pages = ls_key_dict[ls_keyword]['page_numbers']

	if pages:
		if pages_numbers_only:
			return pages
		keyword_dict = get_pdf_dict(lsdyna_manual_file_path, pages)

	return keyword_dict

def main():
	keyword_dict = get_tables_lskeyword("*AIRBAG_ADIABATIC_GAS_MODEL")
	print(keyword_dict)
	return keyword_dict

kdict = main()
