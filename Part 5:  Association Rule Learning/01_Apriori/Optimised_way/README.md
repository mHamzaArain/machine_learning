This code is taken from this [link](https://github.com/ymoch/apyori) whichi which is under MIT Liecense.

Module Features
---------------

- Consisted of only one file and depends on no other libraries,
  which enable you to use it portably.
- Able to used as APIs.

Application Features
--------------------

- Supports a JSON output format.
- Supports a TSV output format for 2-items relations.


Installation
------------

Choose one from the following.

- Install with pip :code:`pip install apyori`.
- Put *apyori.py* into your project.
- Run :code:`python setup.py install`.


CLI Usage
---------

First, prepare input data as tab-separated transactions.

- Each item is separated with a tab.
- Each transactions is separated with a line feed code.

Second, run the application.
Input data is given as a standard input or file paths.

- Run with :code:`python apyori.py` command.
- If installed, you can also run with :code:`apyori-run` command.

Input Data
**********

Input data is separated by a tab if tsv used


Basic usage
***********

.. code-block:: shell

    apyori-run < input_tab_separated_data1.tsv


Use TSV output
**************

.. code-block:: shell

    apyori-run -f tsv < input_tab_separated_data.tsv

Fields of output mean:

- Base item.
- Appended item.
- Support.
- Confidence.
- Lift.


Specify the minimum support
***************************

.. code-block:: shell

    apyori-run -s 0.5 < input_tab_separated_data.tsv


Specify the minimum confidence
******************************

.. code-block:: shell

    apyori-run -c 0.5 < input_tab_separated_data.tsv

For more details, use '-h' option. 
