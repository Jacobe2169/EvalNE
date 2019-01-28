EvalNE - A Framework for Evaluating Network Embeddings on Link Prediction
=========================================================================

.. image:: EvalNE-logo.jpg
    :width: 220px
    :alt: EvalNE logo
    :align: center

EvalNE is an open source Python library designed for assessing and comparing the performance of Network Embedding (NE) methods on Link Prediction (LP) tasks. The library intends to simplify this complex and time consuming evaluation process by providing automation and abstraction of tasks such as model hyper-parameter tuning, selection of train and test edges, negative edge sampling and selection of the evaluation metrics, among many others.
EvalNE can be used both as a command line tool and as an API and is compatible with Python 2 and Python 3. 

EvalNE is provided under the MIT_ free software licence and is maintained by Alexandru Mara (alexandru(dot)mara(at)ugent(dot)be). The source code can be found on GitHub_ and BitBucket_. 

.. _MIT: https://opensource.org/licenses/MIT
.. _GitHub: https://github.com/Dru-Mara/EvalNE
.. _BitBucket: https://bitbucket.org/ghentdatascience/evalne/src/master/


See :doc:`the quickstart <quickstart>` to get started.

.. toctree::
   :maxdepth: 2
   :caption: General

   description

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   api

.. toctree::
   :maxdepth: 1
   :caption: More

   license
   acknowledgements
   help


Citation
--------

If you have found EvaNE usefull in your research, please cite our paper in arXiv_:

.. _arXiv: https://wololololololo

.. code-block:: console

    @misc{xxxxx,
    Author = {Alexandru Mara, Jefrey Lijffijt, Tijl De Bie},
    Title = {EvalNE: A Framework for Evaluating Network Embeddings on Link Prediction},
    Year = {2019},
    Eprint = {xxxxx},
    }
