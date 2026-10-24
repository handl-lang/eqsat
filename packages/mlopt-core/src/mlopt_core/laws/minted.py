"""Shipped minted objects — stored-composite materialization in core.

The discovery pipeline's stored objects (``mlopt_discovery.evidence``)
are `Rewrite`-shaped: a record carries the pattern pair, ``cond`` /
``dspec`` / ``derivation`` data, the ``serializable`` / ``missing_hooks``
honesty flags, and — for a synthesized composite — a ``"guard"`` field:
the ``guard_pats`` re-expression maps plus the two parent names from
which ``mlopt_core.meta._compose_guards`` rebuilds the composed
``check`` / ``derive`` hooks at admit time.  That record IS the object —
the hooks are code, but the *spec* of the guard is data.

A minted object earns the library only through the admission gauntlet
(``evidence.run_gauntlet`` — every stage green, ``usable``).  Shipping it
then means materializing the *same measured object* at import: the
records are embedded verbatim below (``_RECORDS``), and
:func:`materialize` replays the store's ``_rebuild_stored`` semantics
over the embedded table — each ``syn_*`` node rebuilds its pattern from
its record and its guard from the named parents, resolved first against
the embedded records (recursively, so a composite-of-composites
rebuilds) and then against the shipped library by name.

Two deliberate differences from the store path, both louder:

* a parent that resolves short of its declared strength — absent from
  both tables, or a record whose ``missing_hooks`` exceed what the
  transport restored — **raises** rather than degrading to the bare
  pattern: a shipped object may never silently lose its guard;
* the leaf parents are the *library objects themselves*
  (``BATCH_NORM_RULES`` / ``CONV_FOLD_RULES`` members), not records —
  a live ``Rewrite`` is its own full-strength declaration, and the
  composed hooks close over exactly the hooks it was minted against.

The shipped composite is therefore the first library member whose
``missing_hooks`` is non-empty — ``("check", "derive")`` — by class, not
by accident: its guard is the transported premise composition, carried
inspectably as ``guard_pats`` + ``parents`` on the object.  The
serialization census (``tests/test_law_serialize.py``) records this
*transported* class separately from the full-data majority.

The embedded records below are the verbatim ``law_json`` rows
``store_object`` wrote for the conv+BN fold's premise closure of minted
intermediates (``bench/drivers/probes/conv_fold_premise_pay_probe.py`` mint;
``project/retros/minted-composite-default.md`` records the admission
account).  They are frozen artifacts: the composite's guard names them
by minted name, and they are scaffolding — they do not ship as rules.

The second embedded family — :data:`_PRIM_RECORDS` — is a different
minting channel at a different shipping boundary: the
``zoo-prim`` mint's 21 admitted ``prim:synop_*`` folds (graphopt
``bench/drivers/overnight.py --corpus zoo-prim``;
``docs/reports/results_overnight_prim_v2.json``), promoted verbatim
out of the domain-side registry into ``DEFAULT``.  The records are
pattern data only — a ``synop_*`` call is a *name* to core, and the
fold needs nothing more to fire, extract and serialize honestly:
``missing_hooks`` is empty on all 21 (``serializable: true`` —
the plain-record majority, not the transported class).  What does
NOT travel is the denotation: a ``synop_*`` op's meaning is the torch
binding that evaluates the spelled subgraph, and that binding lives
domain-side (``mlopt_discovery.shipped_primitives.register()`` seats
it on the ambient executor table) — core knows the patterns; sinks
supply the semantics, like every op already works.  The op
*declaration* (:func:`~mlopt_core.ir.op_def`) is torch-free and DOES
travel: materializing the folds registers the ``synop_*`` generators,
making them first-class IR citizens wherever the laws are.

The third embedded family — :data:`_PRIM2_RECORDS` — is the same
channel one loop deeper: the ``--prim-bootstrap`` mint's 2 admitted
``prim:synop2_*`` folds (graphopt ``bench/drivers/overnight.py
--prim-bootstrap``; ``docs/reports/results_overnight_prim_bootstrap.json``).
Each fold's LHS is a *synop-carrying skeleton* — a manual shape
spelled through a shipped ``synop_*`` op — and the record therefore
embeds one more verbatim ``law_to_data`` row the depth-1 family does
not need: the definitional ``synop2_*.expand`` premise itself.  For a
depth-1 fold the premise is just ``fold.rhs -> fold.lhs``; for a
depth-2 fold the mint's premise is ``synop2_*(args) -> xpat`` where
``xpat`` is the skeleton *fully re-expanded to manual ops* — the
anchored cert derivation then completes ``rhs -> lhs`` in two steps:
the premise unfolds the call to its manual denotation and the shipped
depth-1 ``prim:synop_*`` fold re-folds the embedded shape to the
skeleton.  The minted ``alpha``/``alpha_fold`` pair records both
faces; the two minted folds share one expanded ``alpha`` (they are
two distinct placements of ``synop_bb8c54c201`` in the same add/mul
tree — different fold paths, hence different laws).

The fourth embedded family — :data:`_PRIM3_RECORDS` — is the
bootstrap mint at depth 3 under the repriced frontier: the
``--prim-deep --prim-bootstrap --prim-vocab`` run's 34 admitted
``prim:synop3_*`` folds (graphopt ``bench/drivers/overnight.py
--corpus zoo-prim``; ``docs/reports/results_overnight_prim_vocab.json``).
Same record shape as depth-2 — the fold LHS is a skeleton spelled
through shipped ``synop_*``/``synop2_*`` ops (``embeds_shipped``),
``alpha_fold``/``alpha`` record the skeleton and the fully-expanded
manual denotation, and the definitional ``synop3_*.expand`` premise
embeds verbatim (its RHS is the expansion, not the skeleton).  The
anchored ``rhs -> lhs`` cert derivation is now up to three steps —
the premise unfolds the call to manual ops, then the shipped
depth-1/depth-2 bank folds re-fold each embedded shape — and the
records add ``embeds``: the same-run minted synops the denotation
nests (the loop-closure provenance the depth-2 run measured).  The
pool's stratified cap kept the 34 deeper-tier mints beside 48
depth-2 rows that stayed unshipped (already-named denotations and
tier duplicates of the shipped surface).
"""

from __future__ import annotations

import json
from typing import Any

from mlopt_core import meta
from mlopt_core.egraph import Rewrite
from mlopt_core.ir import op_def
from mlopt_core.laws import tags
from mlopt_core.laws.serialize import object_from_data
from mlopt_core.rulecache import _dec_binding

__all__ = [
    "SYNOP2_SHIPPED",
    "SYNOP3_SHIPPED",
    "SYNOP_SHIPPED",
    "expand_name",
    "materialize",
    "prim2_folds",
    "prim2_premises",
    "prim2_records",
    "prim3_folds",
    "prim3_premises",
    "prim3_records",
    "prim_fold",
    "prim_folds",
    "prim_premises",
    "prim_records",
    "synop_name",
]

#: The verbatim ``law_json`` records of the minted intermediates the
#: shipped composites descend from, keyed by minted (``syn_*``) name.
#: Leaf parents are NOT here — they resolve against the shipped library
#: by name at materialize time.
_RECORDS: dict[str, Any] = json.loads(
    r"""{
 "syn_batch_norm_expand__conv_channel_shift_sub_sp_6": {
  "bound_norm": "spectral",
  "cert": null,
  "cond": null,
  "derivation": [],
  "dspec": null,
  "error_bound": null,
  "guard": {
   "parents": [
    "batch_norm_expand",
    "conv_channel_shift_sub_sp"
   ],
   "pat1": [
    [
     "$attr:TR",
     {
      "a": false
     }
    ],
    [
     "$attr:M",
     {
      "a": 0.1
     }
    ],
    [
     "$attr:EP",
     {
      "a": 1e-05
     }
    ],
    [
     "$attr:CU",
     {
      "a": false
     }
    ],
    [
     "x",
     {
      "t": {
       "args": [
        {
         "mvar": "v0"
        },
        {
         "mvar": "v1"
        },
        {
         "mvar": "v2"
        }
       ],
       "attrs": {
        "padding": {
         "__tuple__": [
          1,
          1
         ]
        },
        "stride": {
         "__tuple__": [
          1,
          1
         ]
        }
       },
       "op": "conv2d"
      }
     }
    ],
    [
     "w",
     {
      "t": {
       "mvar": "v3"
      }
     }
    ],
    [
     "b",
     {
      "t": {
       "mvar": "v4"
      }
     }
    ],
    [
     "rm",
     {
      "t": {
       "mvar": "v5"
      }
     }
    ],
    [
     "rv",
     {
      "t": {
       "mvar": "v6"
      }
     }
    ]
   ],
   "pat2": [
    [
     "$attr:ST",
     {
      "a": {
       "__tuple__": [
        1,
        1
       ]
      }
     }
    ],
    [
     "$attr:PD",
     {
      "a": {
       "__tuple__": [
        1,
        1
       ]
      }
     }
    ],
    [
     "x",
     {
      "t": {
       "mvar": "v0"
      }
     }
    ],
    [
     "W",
     {
      "t": {
       "mvar": "v1"
      }
     }
    ],
    [
     "b",
     {
      "t": {
       "mvar": "v2"
      }
     }
    ],
    [
     "cv",
     {
      "t": {
       "args": [
        {
         "mvar": "v5"
        }
       ],
       "attrs": {
        "shape": "@1:CH"
       },
       "op": "reshape"
      }
     }
    ]
   ]
  },
  "kind": "abstraction",
  "law": "synthesized: batch_norm_expand then conv_channel_shift_sub_sp (seed)",
  "lhs": {
   "args": [
    {
     "args": [
      {
       "mvar": "v0"
      },
      {
       "mvar": "v1"
      },
      {
       "mvar": "v2"
      }
     ],
     "attrs": {
      "padding": {
       "__tuple__": [
        1,
        1
       ]
      },
      "stride": {
       "__tuple__": [
        1,
        1
       ]
      }
     },
     "op": "conv2d"
    },
    {
     "mvar": "v3"
    },
    {
     "mvar": "v4"
    },
    {
     "mvar": "v5"
    },
    {
     "mvar": "v6"
    }
   ],
   "attrs": {
    "cudnn_enabled": false,
    "eps": 1e-05,
    "momentum": 0.1,
    "training": false
   },
   "op": "batch_norm"
  },
  "missing_hooks": [
   "check",
   "derive"
  ],
  "name": "syn_batch_norm_expand__conv_channel_shift_sub_sp_6",
  "rhs": {
   "args": [
    {
     "args": [
      {
       "args": [
        {
         "mvar": "v3"
        }
       ],
       "attrs": {
        "shape": "@1:CH"
       },
       "op": "reshape"
      },
      {
       "args": [
        {
         "args": [
          {
           "mvar": "v0"
          },
          {
           "mvar": "v1"
          },
          {
           "args": [
            {
             "mvar": "v2"
            },
            {
             "args": [
              {
               "args": [
                {
                 "mvar": "v5"
                }
               ],
               "attrs": {
                "shape": "@1:CH"
               },
               "op": "reshape"
              }
             ],
             "attrs": {
              "shape": "@2:SV"
             },
             "op": "reshape"
            }
           ],
           "attrs": {},
           "op": "sub"
          }
         ],
         "attrs": {
          "padding": {
           "__tuple__": [
            1,
            1
           ]
          },
          "stride": {
           "__tuple__": [
            1,
            1
           ]
          }
         },
         "op": "conv2d"
        },
        {
         "args": [
          {
           "args": [
            {
             "args": [
              {
               "mvar": "v6"
              }
             ],
             "attrs": {
              "shape": "@1:CH"
             },
             "op": "reshape"
            },
            {
             "mvar": "@1:EPS"
            }
           ],
           "attrs": {},
           "op": "add"
          }
         ],
         "attrs": {},
         "op": "rsqrt"
        }
       ],
       "attrs": {},
       "op": "mul"
      }
     ],
     "attrs": {},
     "op": "mul"
    },
    {
     "args": [
      {
       "mvar": "v4"
      }
     ],
     "attrs": {
      "shape": "@1:CH"
     },
     "op": "reshape"
    }
   ],
   "attrs": {},
   "op": "add"
  },
  "serializable": false,
  "tags": [],
  "version": 2
 },
 "syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48": {
  "bound_norm": "spectral",
  "cert": null,
  "cond": null,
  "derivation": [],
  "dspec": null,
  "error_bound": null,
  "guard": {
   "parents": [
    "syn_batch_norm_expand__conv_channel_shift_sub_sp_6",
    "conv_channel_scale_sp"
   ],
   "pat1": [
    [
     "v0",
     {
      "t": {
       "mvar": "v0"
      }
     }
    ],
    [
     "v1",
     {
      "t": {
       "mvar": "v1"
      }
     }
    ],
    [
     "v2",
     {
      "t": {
       "mvar": "v2"
      }
     }
    ],
    [
     "v3",
     {
      "t": {
       "mvar": "v3"
      }
     }
    ],
    [
     "v4",
     {
      "t": {
       "mvar": "v4"
      }
     }
    ],
    [
     "v5",
     {
      "t": {
       "mvar": "v5"
      }
     }
    ],
    [
     "v6",
     {
      "t": {
       "mvar": "v6"
      }
     }
    ]
   ],
   "pat2": [
    [
     "$attr:ST",
     {
      "a": {
       "__tuple__": [
        1,
        1
       ]
      }
     }
    ],
    [
     "$attr:PD",
     {
      "a": {
       "__tuple__": [
        1,
        1
       ]
      }
     }
    ],
    [
     "x",
     {
      "t": {
       "mvar": "v0"
      }
     }
    ],
    [
     "W",
     {
      "t": {
       "mvar": "v1"
      }
     }
    ],
    [
     "b",
     {
      "t": {
       "args": [
        {
         "mvar": "v2"
        },
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v5"
            }
           ],
           "attrs": {
            "shape": "@1:@1:CH"
           },
           "op": "reshape"
          }
         ],
         "attrs": {
          "shape": "@1:@2:SV"
         },
         "op": "reshape"
        }
       ],
       "attrs": {},
       "op": "sub"
      }
     }
    ],
    [
     "sv",
     {
      "t": {
       "args": [
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v6"
            }
           ],
           "attrs": {
            "shape": "@1:@1:CH"
           },
           "op": "reshape"
          },
          {
           "mvar": "@1:@1:EPS"
          }
         ],
         "attrs": {},
         "op": "add"
        }
       ],
       "attrs": {},
       "op": "rsqrt"
      }
     }
    ]
   ]
  },
  "kind": "abstraction",
  "law": "synthesized: syn_batch_norm_expand__conv_channel_shift_sub_sp_6 then conv_channel_scale_sp (seed)",
  "lhs": {
   "args": [
    {
     "args": [
      {
       "mvar": "v0"
      },
      {
       "mvar": "v1"
      },
      {
       "mvar": "v2"
      }
     ],
     "attrs": {
      "padding": {
       "__tuple__": [
        1,
        1
       ]
      },
      "stride": {
       "__tuple__": [
        1,
        1
       ]
      }
     },
     "op": "conv2d"
    },
    {
     "mvar": "v3"
    },
    {
     "mvar": "v4"
    },
    {
     "mvar": "v5"
    },
    {
     "mvar": "v6"
    }
   ],
   "attrs": {
    "cudnn_enabled": false,
    "eps": 1e-05,
    "momentum": 0.1,
    "training": false
   },
   "op": "batch_norm"
  },
  "missing_hooks": [
   "check",
   "derive"
  ],
  "name": "syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48",
  "rhs": {
   "args": [
    {
     "args": [
      {
       "args": [
        {
         "mvar": "v3"
        }
       ],
       "attrs": {
        "shape": "@1:@1:CH"
       },
       "op": "reshape"
      },
      {
       "args": [
        {
         "mvar": "v0"
        },
        {
         "args": [
          {
           "mvar": "v1"
          },
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "args": [
                  {
                   "mvar": "v6"
                  }
                 ],
                 "attrs": {
                  "shape": "@1:@1:CH"
                 },
                 "op": "reshape"
                },
                {
                 "mvar": "@1:@1:EPS"
                }
               ],
               "attrs": {},
               "op": "add"
              }
             ],
             "attrs": {},
             "op": "rsqrt"
            }
           ],
           "attrs": {
            "shape": "@2:WV"
           },
           "op": "reshape"
          }
         ],
         "attrs": {},
         "op": "mul"
        },
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v2"
            },
            {
             "args": [
              {
               "args": [
                {
                 "mvar": "v5"
                }
               ],
               "attrs": {
                "shape": "@1:@1:CH"
               },
               "op": "reshape"
              }
             ],
             "attrs": {
              "shape": "@1:@2:SV"
             },
             "op": "reshape"
            }
           ],
           "attrs": {},
           "op": "sub"
          },
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "args": [
                  {
                   "mvar": "v6"
                  }
                 ],
                 "attrs": {
                  "shape": "@1:@1:CH"
                 },
                 "op": "reshape"
                },
                {
                 "mvar": "@1:@1:EPS"
                }
               ],
               "attrs": {},
               "op": "add"
              }
             ],
             "attrs": {},
             "op": "rsqrt"
            }
           ],
           "attrs": {
            "shape": "@2:SV"
           },
           "op": "reshape"
          }
         ],
         "attrs": {},
         "op": "mul"
        }
       ],
       "attrs": {
        "padding": {
         "__tuple__": [
          1,
          1
         ]
        },
        "stride": {
         "__tuple__": [
          1,
          1
         ]
        }
       },
       "op": "conv2d"
      }
     ],
     "attrs": {},
     "op": "mul"
    },
    {
     "args": [
      {
       "mvar": "v4"
      }
     ],
     "attrs": {
      "shape": "@1:@1:CH"
     },
     "op": "reshape"
    }
   ],
   "attrs": {},
   "op": "add"
  },
  "serializable": false,
  "tags": [],
  "version": 2
 },
 "syn_syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48__conv_channel_scale_l_sp_209": {
  "bound_norm": "spectral",
  "cert": null,
  "cond": null,
  "derivation": [],
  "dspec": null,
  "error_bound": null,
  "guard": {
   "parents": [
    "syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48",
    "conv_channel_scale_l_sp"
   ],
   "pat1": [
    [
     "v0",
     {
      "t": {
       "mvar": "v0"
      }
     }
    ],
    [
     "v1",
     {
      "t": {
       "mvar": "v1"
      }
     }
    ],
    [
     "v2",
     {
      "t": {
       "mvar": "v2"
      }
     }
    ],
    [
     "v3",
     {
      "t": {
       "mvar": "v3"
      }
     }
    ],
    [
     "v4",
     {
      "t": {
       "mvar": "v4"
      }
     }
    ],
    [
     "v5",
     {
      "t": {
       "mvar": "v5"
      }
     }
    ],
    [
     "v6",
     {
      "t": {
       "mvar": "v6"
      }
     }
    ]
   ],
   "pat2": [
    [
     "sv",
     {
      "t": {
       "args": [
        {
         "mvar": "v3"
        }
       ],
       "attrs": {
        "shape": "@1:@1:@1:CH"
       },
       "op": "reshape"
      }
     }
    ],
    [
     "$attr:ST",
     {
      "a": {
       "__tuple__": [
        1,
        1
       ]
      }
     }
    ],
    [
     "$attr:PD",
     {
      "a": {
       "__tuple__": [
        1,
        1
       ]
      }
     }
    ],
    [
     "x",
     {
      "t": {
       "mvar": "v0"
      }
     }
    ],
    [
     "W",
     {
      "t": {
       "args": [
        {
         "mvar": "v1"
        },
        {
         "args": [
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "mvar": "v6"
                }
               ],
               "attrs": {
                "shape": "@1:@1:@1:CH"
               },
               "op": "reshape"
              },
              {
               "mvar": "@1:@1:@1:EPS"
              }
             ],
             "attrs": {},
             "op": "add"
            }
           ],
           "attrs": {},
           "op": "rsqrt"
          }
         ],
         "attrs": {
          "shape": "@1:@2:WV"
         },
         "op": "reshape"
        }
       ],
       "attrs": {},
       "op": "mul"
      }
     }
    ],
    [
     "b",
     {
      "t": {
       "args": [
        {
         "args": [
          {
           "mvar": "v2"
          },
          {
           "args": [
            {
             "args": [
              {
               "mvar": "v5"
              }
             ],
             "attrs": {
              "shape": "@1:@1:@1:CH"
             },
             "op": "reshape"
            }
           ],
           "attrs": {
            "shape": "@1:@1:@2:SV"
           },
           "op": "reshape"
          }
         ],
         "attrs": {},
         "op": "sub"
        },
        {
         "args": [
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "mvar": "v6"
                }
               ],
               "attrs": {
                "shape": "@1:@1:@1:CH"
               },
               "op": "reshape"
              },
              {
               "mvar": "@1:@1:@1:EPS"
              }
             ],
             "attrs": {},
             "op": "add"
            }
           ],
           "attrs": {},
           "op": "rsqrt"
          }
         ],
         "attrs": {
          "shape": "@1:@2:SV"
         },
         "op": "reshape"
        }
       ],
       "attrs": {},
       "op": "mul"
      }
     }
    ]
   ]
  },
  "kind": "abstraction",
  "law": "synthesized: syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48 then conv_channel_scale_l_sp (seed)",
  "lhs": {
   "args": [
    {
     "args": [
      {
       "mvar": "v0"
      },
      {
       "mvar": "v1"
      },
      {
       "mvar": "v2"
      }
     ],
     "attrs": {
      "padding": {
       "__tuple__": [
        1,
        1
       ]
      },
      "stride": {
       "__tuple__": [
        1,
        1
       ]
      }
     },
     "op": "conv2d"
    },
    {
     "mvar": "v3"
    },
    {
     "mvar": "v4"
    },
    {
     "mvar": "v5"
    },
    {
     "mvar": "v6"
    }
   ],
   "attrs": {
    "cudnn_enabled": false,
    "eps": 1e-05,
    "momentum": 0.1,
    "training": false
   },
   "op": "batch_norm"
  },
  "missing_hooks": [
   "check",
   "derive"
  ],
  "name": "syn_syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48__conv_channel_scale_l_sp_209",
  "rhs": {
   "args": [
    {
     "args": [
      {
       "mvar": "v0"
      },
      {
       "args": [
        {
         "args": [
          {
           "mvar": "v1"
          },
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "args": [
                  {
                   "mvar": "v6"
                  }
                 ],
                 "attrs": {
                  "shape": "@1:@1:@1:CH"
                 },
                 "op": "reshape"
                },
                {
                 "mvar": "@1:@1:@1:EPS"
                }
               ],
               "attrs": {},
               "op": "add"
              }
             ],
             "attrs": {},
             "op": "rsqrt"
            }
           ],
           "attrs": {
            "shape": "@1:@2:WV"
           },
           "op": "reshape"
          }
         ],
         "attrs": {},
         "op": "mul"
        },
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v3"
            }
           ],
           "attrs": {
            "shape": "@1:@1:@1:CH"
           },
           "op": "reshape"
          }
         ],
         "attrs": {
          "shape": "@2:WV"
         },
         "op": "reshape"
        }
       ],
       "attrs": {},
       "op": "mul"
      },
      {
       "args": [
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v2"
            },
            {
             "args": [
              {
               "args": [
                {
                 "mvar": "v5"
                }
               ],
               "attrs": {
                "shape": "@1:@1:@1:CH"
               },
               "op": "reshape"
              }
             ],
             "attrs": {
              "shape": "@1:@1:@2:SV"
             },
             "op": "reshape"
            }
           ],
           "attrs": {},
           "op": "sub"
          },
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "args": [
                  {
                   "mvar": "v6"
                  }
                 ],
                 "attrs": {
                  "shape": "@1:@1:@1:CH"
                 },
                 "op": "reshape"
                },
                {
                 "mvar": "@1:@1:@1:EPS"
                }
               ],
               "attrs": {},
               "op": "add"
              }
             ],
             "attrs": {},
             "op": "rsqrt"
            }
           ],
           "attrs": {
            "shape": "@1:@2:SV"
           },
           "op": "reshape"
          }
         ],
         "attrs": {},
         "op": "mul"
        },
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v3"
            }
           ],
           "attrs": {
            "shape": "@1:@1:@1:CH"
           },
           "op": "reshape"
          }
         ],
         "attrs": {
          "shape": "@2:SV"
         },
         "op": "reshape"
        }
       ],
       "attrs": {},
       "op": "mul"
      }
     ],
     "attrs": {
      "padding": {
       "__tuple__": [
        1,
        1
       ]
      },
      "stride": {
       "__tuple__": [
        1,
        1
       ]
      }
     },
     "op": "conv2d"
    },
    {
     "args": [
      {
       "mvar": "v4"
      }
     ],
     "attrs": {
      "shape": "@1:@1:@1:CH"
     },
     "op": "reshape"
    }
   ],
   "attrs": {},
   "op": "add"
  },
  "serializable": false,
  "tags": [],
  "version": 2
 },
 "syn_syn_syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48__conv_channel_scale_l_sp_209__conv_channel_shift_sp_896": {
  "bound_norm": "spectral",
  "cert": null,
  "cond": null,
  "derivation": [],
  "dspec": null,
  "error_bound": null,
  "guard": {
   "parents": [
    "syn_syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48__conv_channel_scale_l_sp_209",
    "conv_channel_shift_sp"
   ],
   "pat1": [
    [
     "v0",
     {
      "t": {
       "mvar": "v0"
      }
     }
    ],
    [
     "v1",
     {
      "t": {
       "mvar": "v1"
      }
     }
    ],
    [
     "v2",
     {
      "t": {
       "mvar": "v2"
      }
     }
    ],
    [
     "v3",
     {
      "t": {
       "mvar": "v3"
      }
     }
    ],
    [
     "v4",
     {
      "t": {
       "mvar": "v4"
      }
     }
    ],
    [
     "v5",
     {
      "t": {
       "mvar": "v5"
      }
     }
    ],
    [
     "v6",
     {
      "t": {
       "mvar": "v6"
      }
     }
    ]
   ],
   "pat2": [
    [
     "$attr:ST",
     {
      "a": {
       "__tuple__": [
        1,
        1
       ]
      }
     }
    ],
    [
     "$attr:PD",
     {
      "a": {
       "__tuple__": [
        1,
        1
       ]
      }
     }
    ],
    [
     "x",
     {
      "t": {
       "mvar": "v0"
      }
     }
    ],
    [
     "W",
     {
      "t": {
       "args": [
        {
         "args": [
          {
           "mvar": "v1"
          },
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "args": [
                  {
                   "mvar": "v6"
                  }
                 ],
                 "attrs": {
                  "shape": "@1:@1:@1:@1:CH"
                 },
                 "op": "reshape"
                },
                {
                 "mvar": "@1:@1:@1:@1:EPS"
                }
               ],
               "attrs": {},
               "op": "add"
              }
             ],
             "attrs": {},
             "op": "rsqrt"
            }
           ],
           "attrs": {
            "shape": "@1:@1:@2:WV"
           },
           "op": "reshape"
          }
         ],
         "attrs": {},
         "op": "mul"
        },
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v3"
            }
           ],
           "attrs": {
            "shape": "@1:@1:@1:@1:CH"
           },
           "op": "reshape"
          }
         ],
         "attrs": {
          "shape": "@1:@2:WV"
         },
         "op": "reshape"
        }
       ],
       "attrs": {},
       "op": "mul"
      }
     }
    ],
    [
     "b",
     {
      "t": {
       "args": [
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v2"
            },
            {
             "args": [
              {
               "args": [
                {
                 "mvar": "v5"
                }
               ],
               "attrs": {
                "shape": "@1:@1:@1:@1:CH"
               },
               "op": "reshape"
              }
             ],
             "attrs": {
              "shape": "@1:@1:@1:@2:SV"
             },
             "op": "reshape"
            }
           ],
           "attrs": {},
           "op": "sub"
          },
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "args": [
                  {
                   "mvar": "v6"
                  }
                 ],
                 "attrs": {
                  "shape": "@1:@1:@1:@1:CH"
                 },
                 "op": "reshape"
                },
                {
                 "mvar": "@1:@1:@1:@1:EPS"
                }
               ],
               "attrs": {},
               "op": "add"
              }
             ],
             "attrs": {},
             "op": "rsqrt"
            }
           ],
           "attrs": {
            "shape": "@1:@1:@2:SV"
           },
           "op": "reshape"
          }
         ],
         "attrs": {},
         "op": "mul"
        },
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v3"
            }
           ],
           "attrs": {
            "shape": "@1:@1:@1:@1:CH"
           },
           "op": "reshape"
          }
         ],
         "attrs": {
          "shape": "@1:@2:SV"
         },
         "op": "reshape"
        }
       ],
       "attrs": {},
       "op": "mul"
      }
     }
    ],
    [
     "cv",
     {
      "t": {
       "args": [
        {
         "mvar": "v4"
        }
       ],
       "attrs": {
        "shape": "@1:@1:@1:@1:CH"
       },
       "op": "reshape"
      }
     }
    ]
   ]
  },
  "kind": "abstraction",
  "law": "synthesized: syn_syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48__conv_channel_scale_l_sp_209 then conv_channel_shift_sp (seed)",
  "lhs": {
   "args": [
    {
     "args": [
      {
       "mvar": "v0"
      },
      {
       "mvar": "v1"
      },
      {
       "mvar": "v2"
      }
     ],
     "attrs": {
      "padding": {
       "__tuple__": [
        1,
        1
       ]
      },
      "stride": {
       "__tuple__": [
        1,
        1
       ]
      }
     },
     "op": "conv2d"
    },
    {
     "mvar": "v3"
    },
    {
     "mvar": "v4"
    },
    {
     "mvar": "v5"
    },
    {
     "mvar": "v6"
    }
   ],
   "attrs": {
    "cudnn_enabled": false,
    "eps": 1e-05,
    "momentum": 0.1,
    "training": false
   },
   "op": "batch_norm"
  },
  "missing_hooks": [
   "check",
   "derive"
  ],
  "name": "syn_syn_syn_syn_batch_norm_expand__conv_channel_shift_sub_sp_6__conv_channel_scale_sp_48__conv_channel_scale_l_sp_209__conv_channel_shift_sp_896",
  "rhs": {
   "args": [
    {
     "mvar": "v0"
    },
    {
     "args": [
      {
       "args": [
        {
         "mvar": "v1"
        },
        {
         "args": [
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "mvar": "v6"
                }
               ],
               "attrs": {
                "shape": "@1:@1:@1:@1:CH"
               },
               "op": "reshape"
              },
              {
               "mvar": "@1:@1:@1:@1:EPS"
              }
             ],
             "attrs": {},
             "op": "add"
            }
           ],
           "attrs": {},
           "op": "rsqrt"
          }
         ],
         "attrs": {
          "shape": "@1:@1:@2:WV"
         },
         "op": "reshape"
        }
       ],
       "attrs": {},
       "op": "mul"
      },
      {
       "args": [
        {
         "args": [
          {
           "mvar": "v3"
          }
         ],
         "attrs": {
          "shape": "@1:@1:@1:@1:CH"
         },
         "op": "reshape"
        }
       ],
       "attrs": {
        "shape": "@1:@2:WV"
       },
       "op": "reshape"
      }
     ],
     "attrs": {},
     "op": "mul"
    },
    {
     "args": [
      {
       "args": [
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v2"
            },
            {
             "args": [
              {
               "args": [
                {
                 "mvar": "v5"
                }
               ],
               "attrs": {
                "shape": "@1:@1:@1:@1:CH"
               },
               "op": "reshape"
              }
             ],
             "attrs": {
              "shape": "@1:@1:@1:@2:SV"
             },
             "op": "reshape"
            }
           ],
           "attrs": {},
           "op": "sub"
          },
          {
           "args": [
            {
             "args": [
              {
               "args": [
                {
                 "args": [
                  {
                   "mvar": "v6"
                  }
                 ],
                 "attrs": {
                  "shape": "@1:@1:@1:@1:CH"
                 },
                 "op": "reshape"
                },
                {
                 "mvar": "@1:@1:@1:@1:EPS"
                }
               ],
               "attrs": {},
               "op": "add"
              }
             ],
             "attrs": {},
             "op": "rsqrt"
            }
           ],
           "attrs": {
            "shape": "@1:@1:@2:SV"
           },
           "op": "reshape"
          }
         ],
         "attrs": {},
         "op": "mul"
        },
        {
         "args": [
          {
           "args": [
            {
             "mvar": "v3"
            }
           ],
           "attrs": {
            "shape": "@1:@1:@1:@1:CH"
           },
           "op": "reshape"
          }
         ],
         "attrs": {
          "shape": "@1:@2:SV"
         },
         "op": "reshape"
        }
       ],
       "attrs": {},
       "op": "mul"
      },
      {
       "args": [
        {
         "args": [
          {
           "mvar": "v4"
          }
         ],
         "attrs": {
          "shape": "@1:@1:@1:@1:CH"
         },
         "op": "reshape"
        }
       ],
       "attrs": {
        "shape": "@2:SV"
       },
       "op": "reshape"
      }
     ],
     "attrs": {},
     "op": "add"
    }
   ],
   "attrs": {
    "padding": {
     "__tuple__": [
      1,
      1
     ]
    },
    "stride": {
     "__tuple__": [
      1,
      1
     ]
    }
   },
   "op": "conv2d"
  },
  "serializable": false,
  "tags": [],
  "version": 2
 }
}"""
)


# ---------------------------------------------------------------------------
#  The prim family — ``prim:synop_*`` folds, shipped laws-only
# ---------------------------------------------------------------------------

#: The verbatim records of the 21 admitted ``prim:synop_*`` folds —
#: the ``zoo-prim`` mint's shipped set (graphopt
#: ``bench/drivers/overnight.py --corpus zoo-prim``, protocol
#: ``seed=0``/``xbind=2``: 24 minted, 21 admitted by the full
#: gauntlet, all ``machine-original`` —
#: ``docs/reports/results_overnight_prim_v2.json``), promoted
#: verbatim out of the domain-side registry
#: (``mlopt_discovery.shipped_primitives``) into the core bank.
#: Keyed by the minted ``synop_*`` op name — ``sha256(alpha-normal
#: pattern)[:10]``, a content hash, not a coined label.  Each record
#: carries the fold's ``law_to_data`` row (``fold``) plus the mint's
#: own measurements: ``alpha`` (the content the op name hashes),
#: ``mods`` (the ≥2 bound-module gate), ``sites`` / ``equal_sites``
#: (the fp64 denotation check the arm ran — equal everywhere) and
#: ``handler`` (the grammar minter's verdict — ``"minted"`` where a
#: handler body verified, ``"refused"`` where the primitive ships on
#: its definitional denotation alone).  Frozen artifacts: a
#: deterministic re-mint of the pool reproduces them
#: byte-identically.
_PRIM_RECORDS: dict[str, Any] = json.loads(
    r"""{
 "synop_03889d8fe1": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add 'v2', 'v3')), 'v4')), 'v5')",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v201"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v472"
           },
           {
            "args": [
             {
              "mvar": "v484"
             },
             {
              "mvar": "v485"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "mvar": "v307"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_03889d8fe1",
   "rhs": {
    "args": [
     {
      "mvar": "v201"
     },
     {
      "mvar": "v472"
     },
     {
      "mvar": "v484"
     },
     {
      "mvar": "v485"
     },
     {
      "mvar": "v307"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "synop_03889d8fe1"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v201",
     "v472",
     "v484",
     "v485",
     "v307",
     "v14"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v201",
       [
        "add",
        [
         "mul",
         "v472",
         [
          "add",
          "v484",
          "v485"
         ]
        ],
        "v307"
       ]
      ],
      "v14"
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v201",
      [
       "add",
       [
        "mul",
        "v472",
        [
         "add",
         "v484",
         "v485"
        ]
       ],
       "v307"
      ]
     ],
     "v14"
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v484": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v484": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v484": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v484": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_2b8cfd1c90": {
  "alpha": "(add (mul 'v0', (add 'v1', (mul 'v2', 'v3'))), (mul 'v4', 'v5'))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v223"
       },
       {
        "args": [
         {
          "mvar": "v267"
         },
         {
          "args": [
           {
            "mvar": "v347"
           },
           {
            "mvar": "v348"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v215"
       },
       {
        "mvar": "v216"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_2b8cfd1c90",
   "rhs": {
    "args": [
     {
      "mvar": "v223"
     },
     {
      "mvar": "v267"
     },
     {
      "mvar": "v347"
     },
     {
      "mvar": "v348"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     }
    ],
    "attrs": {},
    "op": "synop_2b8cfd1c90"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v223",
     "v267",
     "v347",
     "v348",
     "v215",
     "v216"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v223",
       [
        "add",
        "v267",
        [
         "mul",
         "v347",
         "v348"
        ]
       ]
      ],
      [
       "mul",
       "v215",
       "v216"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v223",
      [
       "add",
       "v267",
       [
        "mul",
        "v347",
        "v348"
       ]
      ]
     ],
     [
      "mul",
      "v215",
      "v216"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v267": [
        2,
        16,
        4
       ],
       "v347": [
        2,
        16,
        1
       ],
       "v348": [
        2,
        1,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v267": [
        2,
        16,
        4
       ],
       "v347": [
        2,
        16,
        1
       ],
       "v348": [
        2,
        1,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v267": [
        2,
        16,
        4
       ],
       "v347": [
        2,
        16,
        1
       ],
       "v348": [
        2,
        1,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v267": [
        2,
        16,
        4
       ],
       "v347": [
        2,
        16,
        1
       ],
       "v348": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_39f76b66f8": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', 'v3'), 'v4')), 'v5')), 'v6')",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v201"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v472"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v509"
               },
               {
                "mvar": "v510"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "mvar": "v485"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "mvar": "v307"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_39f76b66f8",
   "rhs": {
    "args": [
     {
      "mvar": "v201"
     },
     {
      "mvar": "v472"
     },
     {
      "mvar": "v509"
     },
     {
      "mvar": "v510"
     },
     {
      "mvar": "v485"
     },
     {
      "mvar": "v307"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "synop_39f76b66f8"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v201",
     "v472",
     "v509",
     "v510",
     "v485",
     "v307",
     "v14"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v201",
       [
        "add",
        [
         "mul",
         "v472",
         [
          "add",
          [
           "mul",
           "v509",
           "v510"
          ],
          "v485"
         ]
        ],
        "v307"
       ]
      ],
      "v14"
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v201",
      [
       "add",
       [
        "mul",
        "v472",
        [
         "add",
         [
          "mul",
          "v509",
          "v510"
         ],
         "v485"
        ]
       ],
       "v307"
      ]
     ],
     "v14"
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v509": [
        2,
        16,
        4
       ],
       "v510": [
        2,
        16,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v509": [
        2,
        16,
        4
       ],
       "v510": [
        2,
        16,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v509": [
        2,
        16,
        4
       ],
       "v510": [
        2,
        16,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v509": [
        2,
        16,
        4
       ],
       "v510": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_4c5296b3ba": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', 'v2'), 'v3')), (mul 'v4', (unsqueeze 'v5', dim=a0)))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v223"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v345"
           },
           {
            "mvar": "v346"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "mvar": "v268"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v215"
       },
       {
        "args": [
         {
          "mvar": "v355"
         }
        ],
        "attrs": {
         "dim": "a97"
        },
        "op": "unsqueeze"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_4c5296b3ba",
   "rhs": {
    "args": [
     {
      "mvar": "v223"
     },
     {
      "mvar": "v345"
     },
     {
      "mvar": "v346"
     },
     {
      "mvar": "v268"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v355"
     }
    ],
    "attrs": {
     "a97": "a97"
    },
    "op": "synop_4c5296b3ba"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "note": "no candidate verified within budget=256",
   "tried": 256,
   "verdict": "refused"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan"
  ],
  "sites": 16
 },
 "synop_4c657cb2c0": {
  "alpha": "(add 'v0', (mul (unsqueeze 'v1', dim=-1), (unsqueeze (select (linear 'v2', 'v3', 'v4'), dim=1, index=a1), dim=a0)))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v13"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v419"
         }
        ],
        "attrs": {
         "dim": -1
        },
        "op": "unsqueeze"
       },
       {
        "args": [
         {
          "args": [
           {
            "args": [
             {
              "mvar": "v633"
             },
             {
              "mvar": "v634"
             },
             {
              "mvar": "v635"
             }
            ],
            "attrs": {},
            "op": "linear"
           }
          ],
          "attrs": {
           "dim": 1,
           "index": "a155"
          },
          "op": "select"
         }
        ],
        "attrs": {
         "dim": "a75"
        },
        "op": "unsqueeze"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_4c657cb2c0",
   "rhs": {
    "args": [
     {
      "mvar": "v13"
     },
     {
      "mvar": "v419"
     },
     {
      "mvar": "v633"
     },
     {
      "mvar": "v634"
     },
     {
      "mvar": "v635"
     }
    ],
    "attrs": {
     "a155": "a155",
     "a75": "a75"
    },
    "op": "synop_4c657cb2c0"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "note": "no candidate verified within budget=256",
   "tried": 256,
   "verdict": "refused"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan"
  ],
  "sites": 16
 },
 "synop_4f8fefdd38": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add 'v2', 'v3')), 'v4')), (mul 'v5', 'v6'))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v223"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v345"
           },
           {
            "args": [
             {
              "mvar": "v362"
             },
             {
              "mvar": "v363"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "mvar": "v268"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v215"
       },
       {
        "mvar": "v216"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_4f8fefdd38",
   "rhs": {
    "args": [
     {
      "mvar": "v223"
     },
     {
      "mvar": "v345"
     },
     {
      "mvar": "v362"
     },
     {
      "mvar": "v363"
     },
     {
      "mvar": "v268"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     }
    ],
    "attrs": {},
    "op": "synop_4f8fefdd38"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v223",
     "v345",
     "v362",
     "v363",
     "v268",
     "v215",
     "v216"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v223",
       [
        "add",
        [
         "mul",
         "v345",
         [
          "add",
          "v362",
          "v363"
         ]
        ],
        "v268"
       ]
      ],
      [
       "mul",
       "v215",
       "v216"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v223",
      [
       "add",
       [
        "mul",
        "v345",
        [
         "add",
         "v362",
         "v363"
        ]
       ],
       "v268"
      ]
     ],
     [
      "mul",
      "v215",
      "v216"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v268": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v362": [
        2,
        16,
        4
       ],
       "v363": [
        2,
        16,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v268": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v362": [
        2,
        16,
        4
       ],
       "v363": [
        2,
        16,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v268": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v362": [
        2,
        16,
        4
       ],
       "v363": [
        2,
        16,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v268": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v362": [
        2,
        16,
        4
       ],
       "v363": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_5147826430": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add 'v2', 'v3')), (mul 'v4', 'v5'))), 'v6')",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v201"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v472"
           },
           {
            "args": [
             {
              "mvar": "v492"
             },
             {
              "mvar": "v493"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v480"
           },
           {
            "mvar": "v481"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_5147826430",
   "rhs": {
    "args": [
     {
      "mvar": "v201"
     },
     {
      "mvar": "v472"
     },
     {
      "mvar": "v492"
     },
     {
      "mvar": "v493"
     },
     {
      "mvar": "v480"
     },
     {
      "mvar": "v481"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "synop_5147826430"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v201",
     "v472",
     "v492",
     "v493",
     "v480",
     "v481",
     "v14"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v201",
       [
        "add",
        [
         "mul",
         "v472",
         [
          "add",
          "v492",
          "v493"
         ]
        ],
        [
         "mul",
         "v480",
         "v481"
        ]
       ]
      ],
      "v14"
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v201",
      [
       "add",
       [
        "mul",
        "v472",
        [
         "add",
         "v492",
         "v493"
        ]
       ],
       [
        "mul",
        "v480",
        "v481"
       ]
      ]
     ],
     "v14"
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v480": [
        2,
        16,
        1
       ],
       "v481": [
        2,
        1,
        4
       ],
       "v492": [
        2,
        16,
        4
       ],
       "v493": [
        2,
        16,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v480": [
        2,
        16,
        1
       ],
       "v481": [
        2,
        1,
        4
       ],
       "v492": [
        2,
        16,
        4
       ],
       "v493": [
        2,
        16,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v480": [
        2,
        16,
        1
       ],
       "v481": [
        2,
        1,
        4
       ],
       "v492": [
        2,
        16,
        4
       ],
       "v493": [
        2,
        16,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v480": [
        2,
        16,
        1
       ],
       "v481": [
        2,
        1,
        4
       ],
       "v492": [
        2,
        16,
        4
       ],
       "v493": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_571fb768b0": {
  "alpha": "(mul 'v0', (add (mul 'v1', (add 'v2', (mul 'v3', 'v4'))), (mul 'v5', 'v6')))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v733"
         },
         {
          "args": [
           {
            "mvar": "v749"
           },
           {
            "args": [
             {
              "mvar": "v771"
             },
             {
              "mvar": "v772"
             }
            ],
            "attrs": {},
            "op": "mul"
           }
          ],
          "attrs": {},
          "op": "add"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "args": [
         {
          "mvar": "v739"
         },
         {
          "mvar": "v740"
         }
        ],
        "attrs": {},
        "op": "mul"
       }
      ],
      "attrs": {},
      "op": "add"
     }
    ],
    "attrs": {},
    "op": "mul"
   },
   "missing_hooks": [],
   "name": "prim:synop_571fb768b0",
   "rhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "mvar": "v733"
     },
     {
      "mvar": "v749"
     },
     {
      "mvar": "v771"
     },
     {
      "mvar": "v772"
     },
     {
      "mvar": "v739"
     },
     {
      "mvar": "v740"
     }
    ],
    "attrs": {},
    "op": "synop_571fb768b0"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v10",
     "v733",
     "v749",
     "v771",
     "v772",
     "v739",
     "v740"
    ],
    "handler_body": [
     "ew",
     [
      "mul",
      "v10",
      [
       "add",
       [
        "mul",
        "v733",
        [
         "add",
         "v749",
         [
          "mul",
          "v771",
          "v772"
         ]
        ]
       ],
       [
        "mul",
        "v739",
        "v740"
       ]
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "mul",
     "v10",
     [
      "add",
      [
       "mul",
       "v733",
       [
        "add",
        "v749",
        [
         "mul",
         "v771",
         "v772"
        ]
       ]
      ],
      [
       "mul",
       "v739",
       "v740"
      ]
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v749": [
        2,
        16,
        4
       ],
       "v771": [
        2,
        16,
        1
       ],
       "v772": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v749": [
        2,
        16,
        4
       ],
       "v771": [
        2,
        16,
        1
       ],
       "v772": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v749": [
        2,
        16,
        4
       ],
       "v771": [
        2,
        16,
        1
       ],
       "v772": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v749": [
        2,
        16,
        4
       ],
       "v771": [
        2,
        16,
        1
       ],
       "v772": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_5e58f26875": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', 'v2'), (mul 'v3', 'v4'))), (mul 'v5', 'v6'))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v223"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v345"
           },
           {
            "mvar": "v346"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v358"
           },
           {
            "mvar": "v359"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v215"
       },
       {
        "mvar": "v216"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_5e58f26875",
   "rhs": {
    "args": [
     {
      "mvar": "v223"
     },
     {
      "mvar": "v345"
     },
     {
      "mvar": "v346"
     },
     {
      "mvar": "v358"
     },
     {
      "mvar": "v359"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     }
    ],
    "attrs": {},
    "op": "synop_5e58f26875"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v223",
     "v345",
     "v346",
     "v358",
     "v359",
     "v215",
     "v216"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v223",
       [
        "add",
        [
         "mul",
         "v345",
         "v346"
        ],
        [
         "mul",
         "v358",
         "v359"
        ]
       ]
      ],
      [
       "mul",
       "v215",
       "v216"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v223",
      [
       "add",
       [
        "mul",
        "v345",
        "v346"
       ],
       [
        "mul",
        "v358",
        "v359"
       ]
      ]
     ],
     [
      "mul",
      "v215",
      "v216"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v346": [
        2,
        16,
        4
       ],
       "v358": [
        2,
        16,
        1
       ],
       "v359": [
        2,
        1,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v346": [
        2,
        16,
        4
       ],
       "v358": [
        2,
        16,
        1
       ],
       "v359": [
        2,
        1,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v346": [
        2,
        16,
        4
       ],
       "v358": [
        2,
        16,
        1
       ],
       "v359": [
        2,
        1,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v346": [
        2,
        16,
        4
       ],
       "v358": [
        2,
        16,
        1
       ],
       "v359": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_629f9ef13b": {
  "alpha": "(mul 'v0', (add (mul 'v1', (add (mul 'v2', 'v3'), (mul 'v4', 'v5'))), 'v6'))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v733"
         },
         {
          "args": [
           {
            "args": [
             {
              "mvar": "v791"
             },
             {
              "mvar": "v792"
             }
            ],
            "attrs": {},
            "op": "mul"
           },
           {
            "args": [
             {
              "mvar": "v801"
             },
             {
              "mvar": "v802"
             }
            ],
            "attrs": {},
            "op": "mul"
           }
          ],
          "attrs": {},
          "op": "add"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "mvar": "v138"
       }
      ],
      "attrs": {},
      "op": "add"
     }
    ],
    "attrs": {},
    "op": "mul"
   },
   "missing_hooks": [],
   "name": "prim:synop_629f9ef13b",
   "rhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "mvar": "v733"
     },
     {
      "mvar": "v791"
     },
     {
      "mvar": "v792"
     },
     {
      "mvar": "v801"
     },
     {
      "mvar": "v802"
     },
     {
      "mvar": "v138"
     }
    ],
    "attrs": {},
    "op": "synop_629f9ef13b"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v10",
     "v733",
     "v791",
     "v792",
     "v801",
     "v802",
     "v138"
    ],
    "handler_body": [
     "ew",
     [
      "mul",
      "v10",
      [
       "add",
       [
        "mul",
        "v733",
        [
         "add",
         [
          "mul",
          "v791",
          "v792"
         ],
         [
          "mul",
          "v801",
          "v802"
         ]
        ]
       ],
       "v138"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "mul",
     "v10",
     [
      "add",
      [
       "mul",
       "v733",
       [
        "add",
        [
         "mul",
         "v791",
         "v792"
        ],
        [
         "mul",
         "v801",
         "v802"
        ]
       ]
      ],
      "v138"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v791": [
        2,
        16,
        4
       ],
       "v792": [
        2,
        16,
        4
       ],
       "v801": [
        2,
        16,
        1
       ],
       "v802": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v791": [
        2,
        16,
        4
       ],
       "v792": [
        2,
        16,
        4
       ],
       "v801": [
        2,
        16,
        1
       ],
       "v802": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v791": [
        2,
        16,
        4
       ],
       "v792": [
        2,
        16,
        4
       ],
       "v801": [
        2,
        16,
        1
       ],
       "v802": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v791": [
        2,
        16,
        4
       ],
       "v792": [
        2,
        16,
        4
       ],
       "v801": [
        2,
        16,
        1
       ],
       "v802": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_62d36aefdb": {
  "alpha": "(mul 'v0', (add (mul 'v1', (add (mul 'v2', 'v3'), 'v4')), (mul 'v5', 'v6')))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v733"
         },
         {
          "args": [
           {
            "args": [
             {
              "mvar": "v769"
             },
             {
              "mvar": "v770"
             }
            ],
            "attrs": {},
            "op": "mul"
           },
           {
            "mvar": "v750"
           }
          ],
          "attrs": {},
          "op": "add"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "args": [
         {
          "mvar": "v739"
         },
         {
          "mvar": "v740"
         }
        ],
        "attrs": {},
        "op": "mul"
       }
      ],
      "attrs": {},
      "op": "add"
     }
    ],
    "attrs": {},
    "op": "mul"
   },
   "missing_hooks": [],
   "name": "prim:synop_62d36aefdb",
   "rhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "mvar": "v733"
     },
     {
      "mvar": "v769"
     },
     {
      "mvar": "v770"
     },
     {
      "mvar": "v750"
     },
     {
      "mvar": "v739"
     },
     {
      "mvar": "v740"
     }
    ],
    "attrs": {},
    "op": "synop_62d36aefdb"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v10",
     "v733",
     "v769",
     "v770",
     "v750",
     "v739",
     "v740"
    ],
    "handler_body": [
     "ew",
     [
      "mul",
      "v10",
      [
       "add",
       [
        "mul",
        "v733",
        [
         "add",
         [
          "mul",
          "v769",
          "v770"
         ],
         "v750"
        ]
       ],
       [
        "mul",
        "v739",
        "v740"
       ]
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "mul",
     "v10",
     [
      "add",
      [
       "mul",
       "v733",
       [
        "add",
        [
         "mul",
         "v769",
         "v770"
        ],
        "v750"
       ]
      ],
      [
       "mul",
       "v739",
       "v740"
      ]
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v750": [
        2,
        16,
        4
       ],
       "v769": [
        2,
        16,
        4
       ],
       "v770": [
        2,
        16,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v750": [
        2,
        16,
        4
       ],
       "v769": [
        2,
        16,
        4
       ],
       "v770": [
        2,
        16,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v750": [
        2,
        16,
        4
       ],
       "v769": [
        2,
        16,
        4
       ],
       "v770": [
        2,
        16,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v750": [
        2,
        16,
        4
       ],
       "v769": [
        2,
        16,
        4
       ],
       "v770": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_6c709f774a": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', 'v2'), (mul 'v3', 'v4'))), 'v5')",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v201"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v472"
           },
           {
            "mvar": "v473"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v480"
           },
           {
            "mvar": "v481"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_6c709f774a",
   "rhs": {
    "args": [
     {
      "mvar": "v201"
     },
     {
      "mvar": "v472"
     },
     {
      "mvar": "v473"
     },
     {
      "mvar": "v480"
     },
     {
      "mvar": "v481"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "synop_6c709f774a"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v201",
     "v472",
     "v473",
     "v480",
     "v481",
     "v14"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v201",
       [
        "add",
        [
         "mul",
         "v472",
         "v473"
        ],
        [
         "mul",
         "v480",
         "v481"
        ]
       ]
      ],
      "v14"
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v201",
      [
       "add",
       [
        "mul",
        "v472",
        "v473"
       ],
       [
        "mul",
        "v480",
        "v481"
       ]
      ]
     ],
     "v14"
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v473": [
        2,
        16,
        4
       ],
       "v480": [
        2,
        16,
        1
       ],
       "v481": [
        2,
        1,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v473": [
        2,
        16,
        4
       ],
       "v480": [
        2,
        16,
        1
       ],
       "v481": [
        2,
        1,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v473": [
        2,
        16,
        4
       ],
       "v480": [
        2,
        16,
        1
       ],
       "v481": [
        2,
        1,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v473": [
        2,
        16,
        4
       ],
       "v480": [
        2,
        16,
        1
       ],
       "v481": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_889d740672": {
  "alpha": "(mul 'v0', (add (mul 'v1', (add 'v2', (mul 'v3', 'v4'))), 'v5'))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v733"
         },
         {
          "args": [
           {
            "mvar": "v743"
           },
           {
            "args": [
             {
              "mvar": "v793"
             },
             {
              "mvar": "v794"
             }
            ],
            "attrs": {},
            "op": "mul"
           }
          ],
          "attrs": {},
          "op": "add"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "mvar": "v138"
       }
      ],
      "attrs": {},
      "op": "add"
     }
    ],
    "attrs": {},
    "op": "mul"
   },
   "missing_hooks": [],
   "name": "prim:synop_889d740672",
   "rhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "mvar": "v733"
     },
     {
      "mvar": "v743"
     },
     {
      "mvar": "v793"
     },
     {
      "mvar": "v794"
     },
     {
      "mvar": "v138"
     }
    ],
    "attrs": {},
    "op": "synop_889d740672"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v10",
     "v733",
     "v743",
     "v793",
     "v794",
     "v138"
    ],
    "handler_body": [
     "ew",
     [
      "mul",
      "v10",
      [
       "add",
       [
        "mul",
        "v733",
        [
         "add",
         "v743",
         [
          "mul",
          "v793",
          "v794"
         ]
        ]
       ],
       "v138"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "mul",
     "v10",
     [
      "add",
      [
       "mul",
       "v733",
       [
        "add",
        "v743",
        [
         "mul",
         "v793",
         "v794"
        ]
       ]
      ],
      "v138"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v743": [
        2,
        16,
        4
       ],
       "v793": [
        2,
        16,
        1
       ],
       "v794": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v743": [
        2,
        16,
        4
       ],
       "v793": [
        2,
        16,
        1
       ],
       "v794": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v743": [
        2,
        16,
        4
       ],
       "v793": [
        2,
        16,
        1
       ],
       "v794": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v743": [
        2,
        16,
        4
       ],
       "v793": [
        2,
        16,
        1
       ],
       "v794": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_93e57c595e": {
  "alpha": "(mul 'v0', (add (mul 'v1', (add (mul 'v2', 'v3'), 'v4')), 'v5'))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v733"
         },
         {
          "args": [
           {
            "args": [
             {
              "mvar": "v791"
             },
             {
              "mvar": "v792"
             }
            ],
            "attrs": {},
            "op": "mul"
           },
           {
            "mvar": "v744"
           }
          ],
          "attrs": {},
          "op": "add"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "mvar": "v138"
       }
      ],
      "attrs": {},
      "op": "add"
     }
    ],
    "attrs": {},
    "op": "mul"
   },
   "missing_hooks": [],
   "name": "prim:synop_93e57c595e",
   "rhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "mvar": "v733"
     },
     {
      "mvar": "v791"
     },
     {
      "mvar": "v792"
     },
     {
      "mvar": "v744"
     },
     {
      "mvar": "v138"
     }
    ],
    "attrs": {},
    "op": "synop_93e57c595e"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v10",
     "v733",
     "v791",
     "v792",
     "v744",
     "v138"
    ],
    "handler_body": [
     "ew",
     [
      "mul",
      "v10",
      [
       "add",
       [
        "mul",
        "v733",
        [
         "add",
         [
          "mul",
          "v791",
          "v792"
         ],
         "v744"
        ]
       ],
       "v138"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "mul",
     "v10",
     [
      "add",
      [
       "mul",
       "v733",
       [
        "add",
        [
         "mul",
         "v791",
         "v792"
        ],
        "v744"
       ]
      ],
      "v138"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v744": [
        2,
        16,
        4
       ],
       "v791": [
        2,
        16,
        4
       ],
       "v792": [
        2,
        16,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v744": [
        2,
        16,
        4
       ],
       "v791": [
        2,
        16,
        4
       ],
       "v792": [
        2,
        16,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v744": [
        2,
        16,
        4
       ],
       "v791": [
        2,
        16,
        4
       ],
       "v792": [
        2,
        16,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v138": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v744": [
        2,
        16,
        4
       ],
       "v791": [
        2,
        16,
        4
       ],
       "v792": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_976f9836d7": {
  "alpha": "(add (mul 'v0', (add 'v1', (mul 'v2', 'v3'))), (mul 'v4', (unsqueeze 'v5', dim=a0)))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v223"
       },
       {
        "args": [
         {
          "mvar": "v267"
         },
         {
          "args": [
           {
            "mvar": "v347"
           },
           {
            "mvar": "v348"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v215"
       },
       {
        "args": [
         {
          "mvar": "v370"
         }
        ],
        "attrs": {
         "dim": "a101"
        },
        "op": "unsqueeze"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_976f9836d7",
   "rhs": {
    "args": [
     {
      "mvar": "v223"
     },
     {
      "mvar": "v267"
     },
     {
      "mvar": "v347"
     },
     {
      "mvar": "v348"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v370"
     }
    ],
    "attrs": {
     "a101": "a101"
    },
    "op": "synop_976f9836d7"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "note": "no candidate verified within budget=256",
   "tried": 256,
   "verdict": "refused"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan"
  ],
  "sites": 16
 },
 "synop_98a002d64b": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', 'v2'), (mul 'v3', (unsqueeze 'v4', dim=a0)))), 'v5')",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v201"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v472"
           },
           {
            "mvar": "v473"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v480"
           },
           {
            "args": [
             {
              "mvar": "v500"
             }
            ],
            "attrs": {
             "dim": "a128"
            },
            "op": "unsqueeze"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_98a002d64b",
   "rhs": {
    "args": [
     {
      "mvar": "v201"
     },
     {
      "mvar": "v472"
     },
     {
      "mvar": "v473"
     },
     {
      "mvar": "v480"
     },
     {
      "mvar": "v500"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {
     "a128": "a128"
    },
    "op": "synop_98a002d64b"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "note": "no candidate verified within budget=256",
   "tried": 256,
   "verdict": "refused"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan"
  ],
  "sites": 16
 },
 "synop_b246dcb838": {
  "alpha": "(mul 'v0', (add (mul 'v1', (add 'v2', 'v3')), (mul 'v4', 'v5')))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v733"
         },
         {
          "args": [
           {
            "mvar": "v749"
           },
           {
            "mvar": "v750"
           }
          ],
          "attrs": {},
          "op": "add"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "args": [
         {
          "mvar": "v739"
         },
         {
          "mvar": "v740"
         }
        ],
        "attrs": {},
        "op": "mul"
       }
      ],
      "attrs": {},
      "op": "add"
     }
    ],
    "attrs": {},
    "op": "mul"
   },
   "missing_hooks": [],
   "name": "prim:synop_b246dcb838",
   "rhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "mvar": "v733"
     },
     {
      "mvar": "v749"
     },
     {
      "mvar": "v750"
     },
     {
      "mvar": "v739"
     },
     {
      "mvar": "v740"
     }
    ],
    "attrs": {},
    "op": "synop_b246dcb838"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v10",
     "v733",
     "v749",
     "v750",
     "v739",
     "v740"
    ],
    "handler_body": [
     "ew",
     [
      "mul",
      "v10",
      [
       "add",
       [
        "mul",
        "v733",
        [
         "add",
         "v749",
         "v750"
        ]
       ],
       [
        "mul",
        "v739",
        "v740"
       ]
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "mul",
     "v10",
     [
      "add",
      [
       "mul",
       "v733",
       [
        "add",
        "v749",
        "v750"
       ]
      ],
      [
       "mul",
       "v739",
       "v740"
      ]
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v749": [
        2,
        16,
        4
       ],
       "v750": [
        2,
        16,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v749": [
        2,
        16,
        4
       ],
       "v750": [
        2,
        16,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v749": [
        2,
        16,
        4
       ],
       "v750": [
        2,
        16,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v733": [
        2,
        16,
        4
       ],
       "v739": [
        2,
        16,
        1
       ],
       "v740": [
        2,
        1,
        4
       ],
       "v749": [
        2,
        16,
        4
       ],
       "v750": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_bb8c54c201": {
  "alpha": "(add (mul 'v0', 'v1'), (mul 'v2', 'v3'))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v223"
       },
       {
        "mvar": "v224"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v215"
       },
       {
        "mvar": "v216"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_bb8c54c201",
   "rhs": {
    "args": [
     {
      "mvar": "v223"
     },
     {
      "mvar": "v224"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     }
    ],
    "attrs": {},
    "op": "synop_bb8c54c201"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v223",
     "v224",
     "v215",
     "v216"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v223",
       "v224"
      ],
      [
       "mul",
       "v215",
       "v216"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v223",
      "v224"
     ],
     [
      "mul",
      "v215",
      "v216"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v215": [
        4,
        16
       ],
       "v216": [
        4,
        16
       ],
       "v223": [
        4,
        16
       ],
       "v224": [
        4,
        16
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v224": [
        2,
        16,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v224": [
        2,
        16,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v224": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "BlockSparseAttention",
   "DecoupledMLA",
   "DilatedBlockAttention",
   "GRUFusedGate",
   "GatedDeltaRule",
   "GatedLinearAttention",
   "HighwayGate",
   "MambaSelectiveScan",
   "PartialRotary",
   "RWKVMixBlock",
   "RealGatedLRU",
   "RotaryPair",
   "SinkWindowAttention",
   "SlidingWindowAttention"
  ],
  "sites": 16
 },
 "synop_d5b128282a": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add 'v2', (mul 'v3', 'v4'))), 'v5')), 'v6')",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v201"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v472"
           },
           {
            "args": [
             {
              "mvar": "v484"
             },
             {
              "args": [
               {
                "mvar": "v511"
               },
               {
                "mvar": "v512"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "mvar": "v307"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_d5b128282a",
   "rhs": {
    "args": [
     {
      "mvar": "v201"
     },
     {
      "mvar": "v472"
     },
     {
      "mvar": "v484"
     },
     {
      "mvar": "v511"
     },
     {
      "mvar": "v512"
     },
     {
      "mvar": "v307"
     },
     {
      "mvar": "v14"
     }
    ],
    "attrs": {},
    "op": "synop_d5b128282a"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v201",
     "v472",
     "v484",
     "v511",
     "v512",
     "v307",
     "v14"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v201",
       [
        "add",
        [
         "mul",
         "v472",
         [
          "add",
          "v484",
          [
           "mul",
           "v511",
           "v512"
          ]
         ]
        ],
        "v307"
       ]
      ],
      "v14"
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v201",
      [
       "add",
       [
        "mul",
        "v472",
        [
         "add",
         "v484",
         [
          "mul",
          "v511",
          "v512"
         ]
        ]
       ],
       "v307"
      ]
     ],
     "v14"
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v484": [
        2,
        16,
        4
       ],
       "v511": [
        2,
        16,
        1
       ],
       "v512": [
        2,
        1,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v484": [
        2,
        16,
        4
       ],
       "v511": [
        2,
        16,
        1
       ],
       "v512": [
        2,
        1,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v484": [
        2,
        16,
        4
       ],
       "v511": [
        2,
        16,
        1
       ],
       "v512": [
        2,
        1,
        4
       ]
      },
      {
       "v14": [
        2,
        16,
        4
       ],
       "v201": [
        2,
        16,
        4
       ],
       "v307": [
        2,
        16,
        4
       ],
       "v472": [
        2,
        16,
        4
       ],
       "v484": [
        2,
        16,
        4
       ],
       "v511": [
        2,
        16,
        1
       ],
       "v512": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop_f08a53f9a0": {
  "alpha": "(add (mul 'v0', (add 'v1', (mul 'v2', (unsqueeze 'v3', dim=a0)))), (mul 'v4', 'v5'))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v223"
       },
       {
        "args": [
         {
          "mvar": "v267"
         },
         {
          "args": [
           {
            "mvar": "v347"
           },
           {
            "args": [
             {
              "mvar": "v381"
             }
            ],
            "attrs": {
             "dim": "a103"
            },
            "op": "unsqueeze"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v215"
       },
       {
        "mvar": "v216"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_f08a53f9a0",
   "rhs": {
    "args": [
     {
      "mvar": "v223"
     },
     {
      "mvar": "v267"
     },
     {
      "mvar": "v347"
     },
     {
      "mvar": "v381"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     }
    ],
    "attrs": {
     "a103": "a103"
    },
    "op": "synop_f08a53f9a0"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "note": "no candidate verified within budget=256",
   "tried": 256,
   "verdict": "refused"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan"
  ],
  "sites": 16
 },
 "synop_f3290a1163": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', 'v2'), 'v3')), (mul 'v4', 'v5'))",
  "equal_sites": 16,
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v223"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v345"
           },
           {
            "mvar": "v346"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "mvar": "v268"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v215"
       },
       {
        "mvar": "v216"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop_f3290a1163",
   "rhs": {
    "args": [
     {
      "mvar": "v223"
     },
     {
      "mvar": "v345"
     },
     {
      "mvar": "v346"
     },
     {
      "mvar": "v268"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     }
    ],
    "attrs": {},
    "op": "synop_f3290a1163"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v223",
     "v345",
     "v346",
     "v268",
     "v215",
     "v216"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v223",
       [
        "add",
        [
         "mul",
         "v345",
         "v346"
        ],
        "v268"
       ]
      ],
      [
       "mul",
       "v215",
       "v216"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v223",
      [
       "add",
       [
        "mul",
        "v345",
        "v346"
       ],
       "v268"
      ]
     ],
     [
      "mul",
      "v215",
      "v216"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 256,
     "domain": [
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v268": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v346": [
        2,
        16,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v268": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v346": [
        2,
        16,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v268": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v346": [
        2,
        16,
        4
       ]
      },
      {
       "v215": [
        2,
        16,
        1
       ],
       "v216": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v268": [
        2,
        16,
        4
       ],
       "v345": [
        2,
        16,
        4
       ],
       "v346": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v2",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 }
}"""
)

#: The ``--prim-bootstrap`` mint's depth-2 records — the 2 admitted
#: ``prim:synop2_*`` folds, verbatim (graphopt
#: ``docs/reports/results_overnight_prim_bootstrap.json``'s
#: ``prim_ops`` rows plus their ``law_to_data`` rows).  Same shape as
#: :data:`_PRIM_RECORDS` plus three depth-2 fields: ``alpha_fold``
#: (the synop-carrying skeleton's alpha-normal key — the fold LHS's
#: spelling, distinct from ``alpha`` which is the fully-expanded
#: manual denotation), ``embeds_shipped`` (the shipped depth-1 ops the
#: skeleton composes over) / ``embeds`` (the pool's loop-closure
#: provenance), and ``expand`` — the definitional premise's own
#: record, embedded because ``expand.rhs`` is the *expanded* manual
#: denotation, not the fold's LHS skeleton.
_PRIM2_RECORDS: dict[str, Any] = json.loads(
    r"""{
 "synop2_35bbd03301": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', 'v4'), (mul 'v5', 'v6'))), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))",
  "alpha_fold": "(add (mul 'v0', (add (mul 'v1', (synop_bb8c54c201 'v2', (add (mul 'v3', 'v4'), (mul 'v5', 'v6')), 'v7', 'v8')), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))",
  "embeds": [
   "synop_077a84bd18"
  ],
  "embeds_shipped": [
   "synop_bb8c54c201"
  ],
  "equal_sites": 16,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop2_35bbd03301 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v485"
     },
     {
      "mvar": "v834"
     },
     {
      "mvar": "v1066"
     },
     {
      "mvar": "v4809"
     },
     {
      "mvar": "v4810"
     },
     {
      "mvar": "v4873"
     },
     {
      "mvar": "v4874"
     },
     {
      "mvar": "v1068"
     },
     {
      "mvar": "v1069"
     },
     {
      "mvar": "v886"
     },
     {
      "mvar": "v887"
     },
     {
      "mvar": "v477"
     },
     {
      "mvar": "v478"
     }
    ],
    "attrs": {},
    "op": "synop2_35bbd03301"
   },
   "missing_hooks": [],
   "name": "synop2_35bbd03301.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v485"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v834"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v1066"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v4809"
                   },
                   {
                    "mvar": "v4810"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v4873"
                   },
                   {
                    "mvar": "v4874"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v1068"
               },
               {
                "mvar": "v1069"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v886"
           },
           {
            "mvar": "v887"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v477"
       },
       {
        "mvar": "v478"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v485"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v834"
           },
           {
            "args": [
             {
              "mvar": "v1066"
             },
             {
              "args": [
               {
                "args": [
                 {
                  "mvar": "v4809"
                 },
                 {
                  "mvar": "v4810"
                 }
                ],
                "attrs": {},
                "op": "mul"
               },
               {
                "args": [
                 {
                  "mvar": "v4873"
                 },
                 {
                  "mvar": "v4874"
                 }
                ],
                "attrs": {},
                "op": "mul"
               }
              ],
              "attrs": {},
              "op": "add"
             },
             {
              "mvar": "v1068"
             },
             {
              "mvar": "v1069"
             }
            ],
            "attrs": {},
            "op": "synop_bb8c54c201"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v886"
           },
           {
            "mvar": "v887"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v477"
       },
       {
        "mvar": "v478"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop2_35bbd03301",
   "rhs": {
    "args": [
     {
      "mvar": "v485"
     },
     {
      "mvar": "v834"
     },
     {
      "mvar": "v1066"
     },
     {
      "mvar": "v4809"
     },
     {
      "mvar": "v4810"
     },
     {
      "mvar": "v4873"
     },
     {
      "mvar": "v4874"
     },
     {
      "mvar": "v1068"
     },
     {
      "mvar": "v1069"
     },
     {
      "mvar": "v886"
     },
     {
      "mvar": "v887"
     },
     {
      "mvar": "v477"
     },
     {
      "mvar": "v478"
     }
    ],
    "attrs": {},
    "op": "synop2_35bbd03301"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v485",
     "v834",
     "v1066",
     "v4809",
     "v4810",
     "v4873",
     "v4874",
     "v1068",
     "v1069",
     "v886",
     "v887",
     "v477",
     "v478"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v485",
       [
        "add",
        [
         "mul",
         "v834",
         [
          "add",
          [
           "mul",
           "v1066",
           [
            "add",
            [
             "mul",
             "v4809",
             "v4810"
            ],
            [
             "mul",
             "v4873",
             "v4874"
            ]
           ]
          ],
          [
           "mul",
           "v1068",
           "v1069"
          ]
         ]
        ],
        [
         "mul",
         "v886",
         "v887"
        ]
       ]
      ],
      [
       "mul",
       "v477",
       "v478"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v485",
      [
       "add",
       [
        "mul",
        "v834",
        [
         "add",
         [
          "mul",
          "v1066",
          [
           "add",
           [
            "mul",
            "v4809",
            "v4810"
           ],
           [
            "mul",
            "v4873",
            "v4874"
           ]
          ]
         ],
         [
          "mul",
          "v1068",
          "v1069"
         ]
        ]
       ],
       [
        "mul",
        "v886",
        "v887"
       ]
      ]
     ],
     [
      "mul",
      "v477",
      "v478"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v1066": [
        2,
        16,
        4
       ],
       "v1068": [
        2,
        16,
        1
       ],
       "v1069": [
        2,
        1,
        4
       ],
       "v477": [
        2,
        16,
        1
       ],
       "v478": [
        2,
        1,
        4
       ],
       "v4809": [
        2,
        16,
        4
       ],
       "v4810": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v4873": [
        2,
        16,
        1
       ],
       "v4874": [
        2,
        1,
        4
       ],
       "v834": [
        2,
        16,
        4
       ],
       "v886": [
        2,
        16,
        1
       ],
       "v887": [
        2,
        1,
        4
       ]
      },
      {
       "v1066": [
        2,
        16,
        4
       ],
       "v1068": [
        2,
        16,
        1
       ],
       "v1069": [
        2,
        1,
        4
       ],
       "v477": [
        2,
        16,
        1
       ],
       "v478": [
        2,
        1,
        4
       ],
       "v4809": [
        2,
        16,
        4
       ],
       "v4810": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v4873": [
        2,
        16,
        1
       ],
       "v4874": [
        2,
        1,
        4
       ],
       "v834": [
        2,
        16,
        4
       ],
       "v886": [
        2,
        16,
        1
       ],
       "v887": [
        2,
        1,
        4
       ]
      },
      {
       "v1066": [
        2,
        16,
        4
       ],
       "v1068": [
        2,
        16,
        1
       ],
       "v1069": [
        2,
        1,
        4
       ],
       "v477": [
        2,
        16,
        1
       ],
       "v478": [
        2,
        1,
        4
       ],
       "v4809": [
        2,
        16,
        4
       ],
       "v4810": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v4873": [
        2,
        16,
        1
       ],
       "v4874": [
        2,
        1,
        4
       ],
       "v834": [
        2,
        16,
        4
       ],
       "v886": [
        2,
        16,
        1
       ],
       "v887": [
        2,
        1,
        4
       ]
      },
      {
       "v1066": [
        2,
        16,
        4
       ],
       "v1068": [
        2,
        16,
        1
       ],
       "v1069": [
        2,
        1,
        4
       ],
       "v477": [
        2,
        16,
        1
       ],
       "v478": [
        2,
        1,
        4
       ],
       "v4809": [
        2,
        16,
        4
       ],
       "v4810": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v4873": [
        2,
        16,
        1
       ],
       "v4874": [
        2,
        1,
        4
       ],
       "v834": [
        2,
        16,
        4
       ],
       "v886": [
        2,
        16,
        1
       ],
       "v887": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v5",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 },
 "synop2_d46808a970": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', 'v4'), (mul 'v5', 'v6'))), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))",
  "alpha_fold": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (synop_bb8c54c201 'v3', 'v4', 'v5', 'v6')), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))",
  "embeds": [
   "synop_077a84bd18"
  ],
  "embeds_shipped": [
   "synop_bb8c54c201"
  ],
  "equal_sites": 16,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop2_d46808a970 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v485"
     },
     {
      "mvar": "v834"
     },
     {
      "mvar": "v1114"
     },
     {
      "mvar": "v1364"
     },
     {
      "mvar": "v1365"
     },
     {
      "mvar": "v1366"
     },
     {
      "mvar": "v1367"
     },
     {
      "mvar": "v1175"
     },
     {
      "mvar": "v1176"
     },
     {
      "mvar": "v886"
     },
     {
      "mvar": "v887"
     },
     {
      "mvar": "v477"
     },
     {
      "mvar": "v478"
     }
    ],
    "attrs": {},
    "op": "synop2_d46808a970"
   },
   "missing_hooks": [],
   "name": "synop2_d46808a970.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v485"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v834"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v1114"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v1364"
                   },
                   {
                    "mvar": "v1365"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v1366"
                   },
                   {
                    "mvar": "v1367"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v1175"
               },
               {
                "mvar": "v1176"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v886"
           },
           {
            "mvar": "v887"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v477"
       },
       {
        "mvar": "v478"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v485"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v834"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v1114"
               },
               {
                "args": [
                 {
                  "mvar": "v1364"
                 },
                 {
                  "mvar": "v1365"
                 },
                 {
                  "mvar": "v1366"
                 },
                 {
                  "mvar": "v1367"
                 }
                ],
                "attrs": {},
                "op": "synop_bb8c54c201"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v1175"
               },
               {
                "mvar": "v1176"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v886"
           },
           {
            "mvar": "v887"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v477"
       },
       {
        "mvar": "v478"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "missing_hooks": [],
   "name": "prim:synop2_d46808a970",
   "rhs": {
    "args": [
     {
      "mvar": "v485"
     },
     {
      "mvar": "v834"
     },
     {
      "mvar": "v1114"
     },
     {
      "mvar": "v1364"
     },
     {
      "mvar": "v1365"
     },
     {
      "mvar": "v1366"
     },
     {
      "mvar": "v1367"
     },
     {
      "mvar": "v1175"
     },
     {
      "mvar": "v1176"
     },
     {
      "mvar": "v886"
     },
     {
      "mvar": "v887"
     },
     {
      "mvar": "v477"
     },
     {
      "mvar": "v478"
     }
    ],
    "attrs": {},
    "op": "synop2_d46808a970"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v485",
     "v834",
     "v1114",
     "v1364",
     "v1365",
     "v1366",
     "v1367",
     "v1175",
     "v1176",
     "v886",
     "v887",
     "v477",
     "v478"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v485",
       [
        "add",
        [
         "mul",
         "v834",
         [
          "add",
          [
           "mul",
           "v1114",
           [
            "add",
            [
             "mul",
             "v1364",
             "v1365"
            ],
            [
             "mul",
             "v1366",
             "v1367"
            ]
           ]
          ],
          [
           "mul",
           "v1175",
           "v1176"
          ]
         ]
        ],
        [
         "mul",
         "v886",
         "v887"
        ]
       ]
      ],
      [
       "mul",
       "v477",
       "v478"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v485",
      [
       "add",
       [
        "mul",
        "v834",
        [
         "add",
         [
          "mul",
          "v1114",
          [
           "add",
           [
            "mul",
            "v1364",
            "v1365"
           ],
           [
            "mul",
            "v1366",
            "v1367"
           ]
          ]
         ],
         [
          "mul",
          "v1175",
          "v1176"
         ]
        ]
       ],
       [
        "mul",
        "v886",
        "v887"
       ]
      ]
     ],
     [
      "mul",
      "v477",
      "v478"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v1114": [
        2,
        16,
        4
       ],
       "v1175": [
        2,
        16,
        1
       ],
       "v1176": [
        2,
        1,
        4
       ],
       "v1364": [
        2,
        16,
        4
       ],
       "v1365": [
        2,
        16,
        4
       ],
       "v1366": [
        2,
        16,
        1
       ],
       "v1367": [
        2,
        1,
        4
       ],
       "v477": [
        2,
        16,
        1
       ],
       "v478": [
        2,
        1,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v834": [
        2,
        16,
        4
       ],
       "v886": [
        2,
        16,
        1
       ],
       "v887": [
        2,
        1,
        4
       ]
      },
      {
       "v1114": [
        2,
        16,
        4
       ],
       "v1175": [
        2,
        16,
        1
       ],
       "v1176": [
        2,
        1,
        4
       ],
       "v1364": [
        2,
        16,
        4
       ],
       "v1365": [
        2,
        16,
        4
       ],
       "v1366": [
        2,
        16,
        1
       ],
       "v1367": [
        2,
        1,
        4
       ],
       "v477": [
        2,
        16,
        1
       ],
       "v478": [
        2,
        1,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v834": [
        2,
        16,
        4
       ],
       "v886": [
        2,
        16,
        1
       ],
       "v887": [
        2,
        1,
        4
       ]
      },
      {
       "v1114": [
        2,
        16,
        4
       ],
       "v1175": [
        2,
        16,
        1
       ],
       "v1176": [
        2,
        1,
        4
       ],
       "v1364": [
        2,
        16,
        4
       ],
       "v1365": [
        2,
        16,
        4
       ],
       "v1366": [
        2,
        16,
        1
       ],
       "v1367": [
        2,
        1,
        4
       ],
       "v477": [
        2,
        16,
        1
       ],
       "v478": [
        2,
        1,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v834": [
        2,
        16,
        4
       ],
       "v886": [
        2,
        16,
        1
       ],
       "v887": [
        2,
        1,
        4
       ]
      },
      {
       "v1114": [
        2,
        16,
        4
       ],
       "v1175": [
        2,
        16,
        1
       ],
       "v1176": [
        2,
        1,
        4
       ],
       "v1364": [
        2,
        16,
        4
       ],
       "v1365": [
        2,
        16,
        4
       ],
       "v1366": [
        2,
        16,
        1
       ],
       "v1367": [
        2,
        1,
        4
       ],
       "v477": [
        2,
        16,
        1
       ],
       "v478": [
        2,
        1,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v834": [
        2,
        16,
        4
       ],
       "v886": [
        2,
        16,
        1
       ],
       "v887": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v5",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedDeltaRule",
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 16
 }
}"""
)

#: The ``--prim-vocab`` run's depth-3 records — the 34 admitted
#: ``prim:synop3_*`` folds, verbatim (graphopt
#: ``docs/reports/results_overnight_prim_vocab.json``'s ``prim_ops``
#: rows plus their ``law_to_data`` rows).  Same shape as
#: :data:`_PRIM2_RECORDS` — ``alpha_fold`` is the synop-carrying
#: skeleton (spelled through shipped ``synop_*``/``synop2_*`` ops,
#: named by ``embeds_shipped``), ``alpha`` the fully-expanded manual
#: denotation, and ``expand`` the embedded definitional premise whose
#: RHS is that expansion.  ``embeds`` is the depth-3 field the
#: depth-2 records did not need as a list (both carried it
#: incidentally): the same-run minted synops the expanded denotation
#: nests — the loop-closure provenance of the vocab-priced frontier.
_PRIM3_RECORDS: dict[str, Any] = json.loads(
    r"""{
 "synop3_04df5fd446": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), 'v7')), (mul 'v8', 'v9'))), (mul 'v10', 'v11'))), (mul 'v12', 'v13'))), (mul 'v14', 'v15'))), (mul 'v16', 'v17'))",
  "alpha_fold": "(synop2_d46808a970 'v0', 'v1', 'v2', 'v3', (synop_f3290a1163 'v4', 'v5', 'v6', 'v7', 'v8', 'v9'), 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17')",
  "embeds": [
   "synop3_f9c72e0a8f"
  ],
  "embeds_shipped": [
   "synop2_d46808a970",
   "synop_f3290a1163"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_04df5fd446 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v713"
     },
     {
      "mvar": "v714"
     },
     {
      "mvar": "v715"
     },
     {
      "mvar": "v716"
     },
     {
      "mvar": "v717"
     },
     {
      "mvar": "v718"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_04df5fd446"
   },
   "missing_hooks": [],
   "name": "synop3_04df5fd446.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v220"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v221"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v222"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v223"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v713"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v714"
                           },
                           {
                            "mvar": "v715"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "mvar": "v716"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v717"
                       },
                       {
                        "mvar": "v718"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v225"
                   },
                   {
                    "mvar": "v226"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v227"
               },
               {
                "mvar": "v228"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v229"
           },
           {
            "mvar": "v230"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v231"
       },
       {
        "mvar": "v232"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "args": [
       {
        "mvar": "v713"
       },
       {
        "mvar": "v714"
       },
       {
        "mvar": "v715"
       },
       {
        "mvar": "v716"
       },
       {
        "mvar": "v717"
       },
       {
        "mvar": "v718"
       }
      ],
      "attrs": {},
      "op": "synop_f3290a1163"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop2_d46808a970"
   },
   "missing_hooks": [],
   "name": "prim:synop3_04df5fd446",
   "rhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v713"
     },
     {
      "mvar": "v714"
     },
     {
      "mvar": "v715"
     },
     {
      "mvar": "v716"
     },
     {
      "mvar": "v717"
     },
     {
      "mvar": "v718"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_04df5fd446"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v220",
     "v221",
     "v222",
     "v223",
     "v713",
     "v714",
     "v715",
     "v716",
     "v717",
     "v718",
     "v225",
     "v226",
     "v227",
     "v228",
     "v229",
     "v230",
     "v231",
     "v232"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v220",
       [
        "add",
        [
         "mul",
         "v221",
         [
          "add",
          [
           "mul",
           "v222",
           [
            "add",
            [
             "mul",
             "v223",
             [
              "add",
              [
               "mul",
               "v713",
               [
                "add",
                [
                 "mul",
                 "v714",
                 "v715"
                ],
                "v716"
               ]
              ],
              [
               "mul",
               "v717",
               "v718"
              ]
             ]
            ],
            [
             "mul",
             "v225",
             "v226"
            ]
           ]
          ],
          [
           "mul",
           "v227",
           "v228"
          ]
         ]
        ],
        [
         "mul",
         "v229",
         "v230"
        ]
       ]
      ],
      [
       "mul",
       "v231",
       "v232"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v220",
      [
       "add",
       [
        "mul",
        "v221",
        [
         "add",
         [
          "mul",
          "v222",
          [
           "add",
           [
            "mul",
            "v223",
            [
             "add",
             [
              "mul",
              "v713",
              [
               "add",
               [
                "mul",
                "v714",
                "v715"
               ],
               "v716"
              ]
             ],
             [
              "mul",
              "v717",
              "v718"
             ]
            ]
           ],
           [
            "mul",
            "v225",
            "v226"
           ]
          ]
         ],
         [
          "mul",
          "v227",
          "v228"
         ]
        ]
       ],
       [
        "mul",
        "v229",
        "v230"
       ]
      ]
     ],
     [
      "mul",
      "v231",
      "v232"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v713": [
        2,
        16,
        4
       ],
       "v714": [
        2,
        16,
        4
       ],
       "v715": [
        2,
        16,
        4
       ],
       "v716": [
        2,
        16,
        4
       ],
       "v717": [
        2,
        16,
        1
       ],
       "v718": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v713": [
        2,
        16,
        4
       ],
       "v714": [
        2,
        16,
        4
       ],
       "v715": [
        2,
        16,
        4
       ],
       "v716": [
        2,
        16,
        4
       ],
       "v717": [
        2,
        16,
        1
       ],
       "v718": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v713": [
        2,
        16,
        4
       ],
       "v714": [
        2,
        16,
        4
       ],
       "v715": [
        2,
        16,
        4
       ],
       "v716": [
        2,
        16,
        4
       ],
       "v717": [
        2,
        16,
        1
       ],
       "v718": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        1
       ],
       "v221": [
        2,
        16,
        1
       ],
       "v222": [
        2,
        16,
        1
       ],
       "v223": [
        2,
        16,
        1
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        16
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        16
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        16
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        16
       ],
       "v713": [
        2,
        16,
        1
       ],
       "v714": [
        2,
        16,
        1
       ],
       "v715": [
        2,
        16,
        16
       ],
       "v716": [
        2,
        16,
        16
       ],
       "v717": [
        2,
        16,
        1
       ],
       "v718": [
        2,
        1,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_083e9c241f": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add 'v4', 'v5')), (mul 'v6', 'v7'))), (mul 'v8', 'v9'))), (mul 'v10', 'v11'))), (mul 'v12', 'v13'))",
  "alpha_fold": "(synop2_35bbd03301 'v0', 'v1', 'v2', 'v3', (add 'v4', 'v5'), 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13')",
  "embeds": [],
  "embeds_shipped": [
   "synop2_35bbd03301"
  ],
  "equal_sites": 12,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_083e9c241f unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v451"
     },
     {
      "mvar": "v452"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_083e9c241f"
   },
   "missing_hooks": [],
   "name": "synop3_083e9c241f.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v207"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v208"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v209"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v210"
                   },
                   {
                    "args": [
                     {
                      "mvar": "v451"
                     },
                     {
                      "mvar": "v452"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v212"
                   },
                   {
                    "mvar": "v213"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v214"
               },
               {
                "mvar": "v215"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v216"
           },
           {
            "mvar": "v217"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v218"
       },
       {
        "mvar": "v219"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "args": [
       {
        "mvar": "v451"
       },
       {
        "mvar": "v452"
       }
      ],
      "attrs": {},
      "op": "add"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop2_35bbd03301"
   },
   "missing_hooks": [],
   "name": "prim:synop3_083e9c241f",
   "rhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v451"
     },
     {
      "mvar": "v452"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_083e9c241f"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v207",
     "v208",
     "v209",
     "v210",
     "v451",
     "v452",
     "v212",
     "v213",
     "v214",
     "v215",
     "v216",
     "v217",
     "v218",
     "v219"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v207",
       [
        "add",
        [
         "mul",
         "v208",
         [
          "add",
          [
           "mul",
           "v209",
           [
            "add",
            [
             "mul",
             "v210",
             [
              "add",
              "v451",
              "v452"
             ]
            ],
            [
             "mul",
             "v212",
             "v213"
            ]
           ]
          ],
          [
           "mul",
           "v214",
           "v215"
          ]
         ]
        ],
        [
         "mul",
         "v216",
         "v217"
        ]
       ]
      ],
      [
       "mul",
       "v218",
       "v219"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v207",
      [
       "add",
       [
        "mul",
        "v208",
        [
         "add",
         [
          "mul",
          "v209",
          [
           "add",
           [
            "mul",
            "v210",
            [
             "add",
             "v451",
             "v452"
            ]
           ],
           [
            "mul",
            "v212",
            "v213"
           ]
          ]
         ],
         [
          "mul",
          "v214",
          "v215"
         ]
        ]
       ],
       [
        "mul",
        "v216",
        "v217"
       ]
      ]
     ],
     [
      "mul",
      "v218",
      "v219"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v451": [
        2,
        16,
        4
       ],
       "v452": [
        2,
        16,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v451": [
        2,
        16,
        4
       ],
       "v452": [
        2,
        16,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v451": [
        2,
        16,
        4
       ],
       "v452": [
        2,
        16,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v451": [
        2,
        16,
        4
       ],
       "v452": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 12
 },
 "synop3_0ca31bf1e5": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add 'v5', (mul 'v6', 'v7'))), (mul 'v8', 'v9'))), (mul 'v10', 'v11'))), (mul 'v12', 'v13'))), (mul 'v14', 'v15'))), (mul 'v16', 'v17'))",
  "alpha_fold": "(synop2_35bbd03301 'v0', 'v1', 'v2', 'v3', (synop_2b8cfd1c90 'v4', 'v5', 'v6', 'v7', 'v8', 'v9'), 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17')",
  "embeds": [
   "synop3_39dbcda988"
  ],
  "embeds_shipped": [
   "synop2_35bbd03301",
   "synop_2b8cfd1c90"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_0ca31bf1e5 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v485"
     },
     {
      "mvar": "v486"
     },
     {
      "mvar": "v487"
     },
     {
      "mvar": "v488"
     },
     {
      "mvar": "v489"
     },
     {
      "mvar": "v490"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_0ca31bf1e5"
   },
   "missing_hooks": [],
   "name": "synop3_0ca31bf1e5.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v207"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v208"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v209"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v210"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v485"
                       },
                       {
                        "args": [
                         {
                          "mvar": "v486"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v487"
                           },
                           {
                            "mvar": "v488"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v489"
                       },
                       {
                        "mvar": "v490"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v212"
                   },
                   {
                    "mvar": "v213"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v214"
               },
               {
                "mvar": "v215"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v216"
           },
           {
            "mvar": "v217"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v218"
       },
       {
        "mvar": "v219"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "args": [
       {
        "mvar": "v485"
       },
       {
        "mvar": "v486"
       },
       {
        "mvar": "v487"
       },
       {
        "mvar": "v488"
       },
       {
        "mvar": "v489"
       },
       {
        "mvar": "v490"
       }
      ],
      "attrs": {},
      "op": "synop_2b8cfd1c90"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop2_35bbd03301"
   },
   "missing_hooks": [],
   "name": "prim:synop3_0ca31bf1e5",
   "rhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v485"
     },
     {
      "mvar": "v486"
     },
     {
      "mvar": "v487"
     },
     {
      "mvar": "v488"
     },
     {
      "mvar": "v489"
     },
     {
      "mvar": "v490"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_0ca31bf1e5"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v207",
     "v208",
     "v209",
     "v210",
     "v485",
     "v486",
     "v487",
     "v488",
     "v489",
     "v490",
     "v212",
     "v213",
     "v214",
     "v215",
     "v216",
     "v217",
     "v218",
     "v219"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v207",
       [
        "add",
        [
         "mul",
         "v208",
         [
          "add",
          [
           "mul",
           "v209",
           [
            "add",
            [
             "mul",
             "v210",
             [
              "add",
              [
               "mul",
               "v485",
               [
                "add",
                "v486",
                [
                 "mul",
                 "v487",
                 "v488"
                ]
               ]
              ],
              [
               "mul",
               "v489",
               "v490"
              ]
             ]
            ],
            [
             "mul",
             "v212",
             "v213"
            ]
           ]
          ],
          [
           "mul",
           "v214",
           "v215"
          ]
         ]
        ],
        [
         "mul",
         "v216",
         "v217"
        ]
       ]
      ],
      [
       "mul",
       "v218",
       "v219"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v207",
      [
       "add",
       [
        "mul",
        "v208",
        [
         "add",
         [
          "mul",
          "v209",
          [
           "add",
           [
            "mul",
            "v210",
            [
             "add",
             [
              "mul",
              "v485",
              [
               "add",
               "v486",
               [
                "mul",
                "v487",
                "v488"
               ]
              ]
             ],
             [
              "mul",
              "v489",
              "v490"
             ]
            ]
           ],
           [
            "mul",
            "v212",
            "v213"
           ]
          ]
         ],
         [
          "mul",
          "v214",
          "v215"
         ]
        ]
       ],
       [
        "mul",
        "v216",
        "v217"
       ]
      ]
     ],
     [
      "mul",
      "v218",
      "v219"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v486": [
        2,
        16,
        4
       ],
       "v487": [
        2,
        16,
        1
       ],
       "v488": [
        2,
        1,
        4
       ],
       "v489": [
        2,
        16,
        1
       ],
       "v490": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v486": [
        2,
        16,
        4
       ],
       "v487": [
        2,
        16,
        1
       ],
       "v488": [
        2,
        1,
        4
       ],
       "v489": [
        2,
        16,
        1
       ],
       "v490": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v486": [
        2,
        16,
        4
       ],
       "v487": [
        2,
        16,
        1
       ],
       "v488": [
        2,
        1,
        4
       ],
       "v489": [
        2,
        16,
        1
       ],
       "v490": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        1
       ],
       "v208": [
        2,
        16,
        1
       ],
       "v209": [
        2,
        16,
        1
       ],
       "v210": [
        2,
        16,
        1
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        16
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        16
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        16
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        16
       ],
       "v485": [
        2,
        16,
        1
       ],
       "v486": [
        2,
        16,
        16
       ],
       "v487": [
        2,
        16,
        1
       ],
       "v488": [
        2,
        1,
        16
       ],
       "v489": [
        2,
        16,
        1
       ],
       "v490": [
        2,
        1,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_0fa19d1ff8": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))), (mul 'v13', 'v14'))), (mul 'v15', 'v16'))), (mul 'v17', 'v18'))",
  "alpha_fold": "(synop_5e58f26875 'v0', 'v1', (synop2_d46808a970 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14'), 'v15', 'v16', 'v17', 'v18')",
  "embeds": [
   "synop3_304457cd8d",
   "synop3_4cdc06b88f"
  ],
  "embeds_shipped": [
   "synop2_d46808a970",
   "synop_5e58f26875"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_0fa19d1ff8 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v115"
     },
     {
      "mvar": "v116"
     },
     {
      "mvar": "v4044"
     },
     {
      "mvar": "v4045"
     },
     {
      "mvar": "v4046"
     },
     {
      "mvar": "v4047"
     },
     {
      "mvar": "v4048"
     },
     {
      "mvar": "v4049"
     },
     {
      "mvar": "v4050"
     },
     {
      "mvar": "v4051"
     },
     {
      "mvar": "v4052"
     },
     {
      "mvar": "v4053"
     },
     {
      "mvar": "v4054"
     },
     {
      "mvar": "v4055"
     },
     {
      "mvar": "v4056"
     },
     {
      "mvar": "v118"
     },
     {
      "mvar": "v119"
     },
     {
      "mvar": "v120"
     },
     {
      "mvar": "v121"
     }
    ],
    "attrs": {},
    "op": "synop3_0fa19d1ff8"
   },
   "missing_hooks": [],
   "name": "synop3_0fa19d1ff8.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v115"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v116"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v4044"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v4045"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v4046"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v4047"
                           },
                           {
                            "mvar": "v4048"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v4049"
                           },
                           {
                            "mvar": "v4050"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v4051"
                       },
                       {
                        "mvar": "v4052"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v4053"
                   },
                   {
                    "mvar": "v4054"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v4055"
               },
               {
                "mvar": "v4056"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v118"
           },
           {
            "mvar": "v119"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v120"
       },
       {
        "mvar": "v121"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v115"
     },
     {
      "mvar": "v116"
     },
     {
      "args": [
       {
        "mvar": "v4044"
       },
       {
        "mvar": "v4045"
       },
       {
        "mvar": "v4046"
       },
       {
        "mvar": "v4047"
       },
       {
        "mvar": "v4048"
       },
       {
        "mvar": "v4049"
       },
       {
        "mvar": "v4050"
       },
       {
        "mvar": "v4051"
       },
       {
        "mvar": "v4052"
       },
       {
        "mvar": "v4053"
       },
       {
        "mvar": "v4054"
       },
       {
        "mvar": "v4055"
       },
       {
        "mvar": "v4056"
       }
      ],
      "attrs": {},
      "op": "synop2_d46808a970"
     },
     {
      "mvar": "v118"
     },
     {
      "mvar": "v119"
     },
     {
      "mvar": "v120"
     },
     {
      "mvar": "v121"
     }
    ],
    "attrs": {},
    "op": "synop_5e58f26875"
   },
   "missing_hooks": [],
   "name": "prim:synop3_0fa19d1ff8",
   "rhs": {
    "args": [
     {
      "mvar": "v115"
     },
     {
      "mvar": "v116"
     },
     {
      "mvar": "v4044"
     },
     {
      "mvar": "v4045"
     },
     {
      "mvar": "v4046"
     },
     {
      "mvar": "v4047"
     },
     {
      "mvar": "v4048"
     },
     {
      "mvar": "v4049"
     },
     {
      "mvar": "v4050"
     },
     {
      "mvar": "v4051"
     },
     {
      "mvar": "v4052"
     },
     {
      "mvar": "v4053"
     },
     {
      "mvar": "v4054"
     },
     {
      "mvar": "v4055"
     },
     {
      "mvar": "v4056"
     },
     {
      "mvar": "v118"
     },
     {
      "mvar": "v119"
     },
     {
      "mvar": "v120"
     },
     {
      "mvar": "v121"
     }
    ],
    "attrs": {},
    "op": "synop3_0fa19d1ff8"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v115",
     "v116",
     "v4044",
     "v4045",
     "v4046",
     "v4047",
     "v4048",
     "v4049",
     "v4050",
     "v4051",
     "v4052",
     "v4053",
     "v4054",
     "v4055",
     "v4056",
     "v118",
     "v119",
     "v120",
     "v121"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v115",
       [
        "add",
        [
         "mul",
         "v116",
         [
          "add",
          [
           "mul",
           "v4044",
           [
            "add",
            [
             "mul",
             "v4045",
             [
              "add",
              [
               "mul",
               "v4046",
               [
                "add",
                [
                 "mul",
                 "v4047",
                 "v4048"
                ],
                [
                 "mul",
                 "v4049",
                 "v4050"
                ]
               ]
              ],
              [
               "mul",
               "v4051",
               "v4052"
              ]
             ]
            ],
            [
             "mul",
             "v4053",
             "v4054"
            ]
           ]
          ],
          [
           "mul",
           "v4055",
           "v4056"
          ]
         ]
        ],
        [
         "mul",
         "v118",
         "v119"
        ]
       ]
      ],
      [
       "mul",
       "v120",
       "v121"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v115",
      [
       "add",
       [
        "mul",
        "v116",
        [
         "add",
         [
          "mul",
          "v4044",
          [
           "add",
           [
            "mul",
            "v4045",
            [
             "add",
             [
              "mul",
              "v4046",
              [
               "add",
               [
                "mul",
                "v4047",
                "v4048"
               ],
               [
                "mul",
                "v4049",
                "v4050"
               ]
              ]
             ],
             [
              "mul",
              "v4051",
              "v4052"
             ]
            ]
           ],
           [
            "mul",
            "v4053",
            "v4054"
           ]
          ]
         ],
         [
          "mul",
          "v4055",
          "v4056"
         ]
        ]
       ],
       [
        "mul",
        "v118",
        "v119"
       ]
      ]
     ],
     [
      "mul",
      "v120",
      "v121"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v115": [
        2,
        16,
        4
       ],
       "v116": [
        2,
        16,
        4
       ],
       "v118": [
        2,
        16,
        1
       ],
       "v119": [
        2,
        1,
        4
       ],
       "v120": [
        2,
        16,
        1
       ],
       "v121": [
        2,
        1,
        4
       ],
       "v4044": [
        2,
        16,
        4
       ],
       "v4045": [
        2,
        16,
        4
       ],
       "v4046": [
        2,
        16,
        4
       ],
       "v4047": [
        2,
        16,
        4
       ],
       "v4048": [
        2,
        16,
        4
       ],
       "v4049": [
        2,
        16,
        1
       ],
       "v4050": [
        2,
        1,
        4
       ],
       "v4051": [
        2,
        16,
        1
       ],
       "v4052": [
        2,
        1,
        4
       ],
       "v4053": [
        2,
        16,
        1
       ],
       "v4054": [
        2,
        1,
        4
       ],
       "v4055": [
        2,
        16,
        1
       ],
       "v4056": [
        2,
        1,
        4
       ]
      },
      {
       "v115": [
        2,
        16,
        4
       ],
       "v116": [
        2,
        16,
        4
       ],
       "v118": [
        2,
        16,
        1
       ],
       "v119": [
        2,
        1,
        4
       ],
       "v120": [
        2,
        16,
        1
       ],
       "v121": [
        2,
        1,
        4
       ],
       "v4044": [
        2,
        16,
        4
       ],
       "v4045": [
        2,
        16,
        4
       ],
       "v4046": [
        2,
        16,
        4
       ],
       "v4047": [
        2,
        16,
        4
       ],
       "v4048": [
        2,
        16,
        4
       ],
       "v4049": [
        2,
        16,
        1
       ],
       "v4050": [
        2,
        1,
        4
       ],
       "v4051": [
        2,
        16,
        1
       ],
       "v4052": [
        2,
        1,
        4
       ],
       "v4053": [
        2,
        16,
        1
       ],
       "v4054": [
        2,
        1,
        4
       ],
       "v4055": [
        2,
        16,
        1
       ],
       "v4056": [
        2,
        1,
        4
       ]
      },
      {
       "v115": [
        2,
        16,
        4
       ],
       "v116": [
        2,
        16,
        4
       ],
       "v118": [
        2,
        16,
        1
       ],
       "v119": [
        2,
        1,
        4
       ],
       "v120": [
        2,
        16,
        1
       ],
       "v121": [
        2,
        1,
        4
       ],
       "v4044": [
        2,
        16,
        4
       ],
       "v4045": [
        2,
        16,
        4
       ],
       "v4046": [
        2,
        16,
        4
       ],
       "v4047": [
        2,
        16,
        4
       ],
       "v4048": [
        2,
        16,
        4
       ],
       "v4049": [
        2,
        16,
        1
       ],
       "v4050": [
        2,
        1,
        4
       ],
       "v4051": [
        2,
        16,
        1
       ],
       "v4052": [
        2,
        1,
        4
       ],
       "v4053": [
        2,
        16,
        1
       ],
       "v4054": [
        2,
        1,
        4
       ],
       "v4055": [
        2,
        16,
        1
       ],
       "v4056": [
        2,
        1,
        4
       ]
      },
      {
       "v115": [
        2,
        16,
        1
       ],
       "v116": [
        2,
        16,
        1
       ],
       "v118": [
        2,
        16,
        1
       ],
       "v119": [
        2,
        1,
        16
       ],
       "v120": [
        2,
        16,
        1
       ],
       "v121": [
        2,
        1,
        16
       ],
       "v4044": [
        2,
        16,
        1
       ],
       "v4045": [
        2,
        16,
        1
       ],
       "v4046": [
        2,
        16,
        1
       ],
       "v4047": [
        2,
        16,
        1
       ],
       "v4048": [
        2,
        16,
        16
       ],
       "v4049": [
        2,
        16,
        1
       ],
       "v4050": [
        2,
        1,
        16
       ],
       "v4051": [
        2,
        16,
        1
       ],
       "v4052": [
        2,
        1,
        16
       ],
       "v4053": [
        2,
        16,
        1
       ],
       "v4054": [
        2,
        1,
        16
       ],
       "v4055": [
        2,
        16,
        1
       ],
       "v4056": [
        2,
        1,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_1850a07696": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add 'v4', 'v5')), (mul 'v6', 'v7'))), (mul 'v8', 'v9'))), (mul 'v10', 'v11'))), (mul 'v12', 'v13'))",
  "alpha_fold": "(synop2_d46808a970 'v0', 'v1', 'v2', 'v3', (add 'v4', 'v5'), 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13')",
  "embeds": [],
  "embeds_shipped": [
   "synop2_d46808a970"
  ],
  "equal_sites": 12,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_1850a07696 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v599"
     },
     {
      "mvar": "v600"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_1850a07696"
   },
   "missing_hooks": [],
   "name": "synop3_1850a07696.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v220"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v221"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v222"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v223"
                   },
                   {
                    "args": [
                     {
                      "mvar": "v599"
                     },
                     {
                      "mvar": "v600"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v225"
                   },
                   {
                    "mvar": "v226"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v227"
               },
               {
                "mvar": "v228"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v229"
           },
           {
            "mvar": "v230"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v231"
       },
       {
        "mvar": "v232"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "args": [
       {
        "mvar": "v599"
       },
       {
        "mvar": "v600"
       }
      ],
      "attrs": {},
      "op": "add"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop2_d46808a970"
   },
   "missing_hooks": [],
   "name": "prim:synop3_1850a07696",
   "rhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v599"
     },
     {
      "mvar": "v600"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_1850a07696"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v220",
     "v221",
     "v222",
     "v223",
     "v599",
     "v600",
     "v225",
     "v226",
     "v227",
     "v228",
     "v229",
     "v230",
     "v231",
     "v232"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v220",
       [
        "add",
        [
         "mul",
         "v221",
         [
          "add",
          [
           "mul",
           "v222",
           [
            "add",
            [
             "mul",
             "v223",
             [
              "add",
              "v599",
              "v600"
             ]
            ],
            [
             "mul",
             "v225",
             "v226"
            ]
           ]
          ],
          [
           "mul",
           "v227",
           "v228"
          ]
         ]
        ],
        [
         "mul",
         "v229",
         "v230"
        ]
       ]
      ],
      [
       "mul",
       "v231",
       "v232"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v220",
      [
       "add",
       [
        "mul",
        "v221",
        [
         "add",
         [
          "mul",
          "v222",
          [
           "add",
           [
            "mul",
            "v223",
            [
             "add",
             "v599",
             "v600"
            ]
           ],
           [
            "mul",
            "v225",
            "v226"
           ]
          ]
         ],
         [
          "mul",
          "v227",
          "v228"
         ]
        ]
       ],
       [
        "mul",
        "v229",
        "v230"
       ]
      ]
     ],
     [
      "mul",
      "v231",
      "v232"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v599": [
        2,
        16,
        4
       ],
       "v600": [
        2,
        16,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v599": [
        2,
        16,
        4
       ],
       "v600": [
        2,
        16,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v599": [
        2,
        16,
        4
       ],
       "v600": [
        2,
        16,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v599": [
        2,
        16,
        4
       ],
       "v600": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 12
 },
 "synop3_207ef0ddb7": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), 'v7')), (mul 'v8', 'v9'))), (mul 'v10', 'v11'))), (mul 'v12', 'v13'))), (mul 'v14', 'v15'))), (mul 'v16', 'v17'))",
  "alpha_fold": "(synop2_35bbd03301 'v0', 'v1', 'v2', 'v3', (synop_f3290a1163 'v4', 'v5', 'v6', 'v7', 'v8', 'v9'), 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17')",
  "embeds": [
   "synop3_f9c72e0a8f"
  ],
  "embeds_shipped": [
   "synop2_35bbd03301",
   "synop_f3290a1163"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_207ef0ddb7 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v565"
     },
     {
      "mvar": "v566"
     },
     {
      "mvar": "v567"
     },
     {
      "mvar": "v568"
     },
     {
      "mvar": "v569"
     },
     {
      "mvar": "v570"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_207ef0ddb7"
   },
   "missing_hooks": [],
   "name": "synop3_207ef0ddb7.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v207"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v208"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v209"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v210"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v565"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v566"
                           },
                           {
                            "mvar": "v567"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "mvar": "v568"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v569"
                       },
                       {
                        "mvar": "v570"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v212"
                   },
                   {
                    "mvar": "v213"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v214"
               },
               {
                "mvar": "v215"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v216"
           },
           {
            "mvar": "v217"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v218"
       },
       {
        "mvar": "v219"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "args": [
       {
        "mvar": "v565"
       },
       {
        "mvar": "v566"
       },
       {
        "mvar": "v567"
       },
       {
        "mvar": "v568"
       },
       {
        "mvar": "v569"
       },
       {
        "mvar": "v570"
       }
      ],
      "attrs": {},
      "op": "synop_f3290a1163"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop2_35bbd03301"
   },
   "missing_hooks": [],
   "name": "prim:synop3_207ef0ddb7",
   "rhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v565"
     },
     {
      "mvar": "v566"
     },
     {
      "mvar": "v567"
     },
     {
      "mvar": "v568"
     },
     {
      "mvar": "v569"
     },
     {
      "mvar": "v570"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_207ef0ddb7"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v207",
     "v208",
     "v209",
     "v210",
     "v565",
     "v566",
     "v567",
     "v568",
     "v569",
     "v570",
     "v212",
     "v213",
     "v214",
     "v215",
     "v216",
     "v217",
     "v218",
     "v219"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v207",
       [
        "add",
        [
         "mul",
         "v208",
         [
          "add",
          [
           "mul",
           "v209",
           [
            "add",
            [
             "mul",
             "v210",
             [
              "add",
              [
               "mul",
               "v565",
               [
                "add",
                [
                 "mul",
                 "v566",
                 "v567"
                ],
                "v568"
               ]
              ],
              [
               "mul",
               "v569",
               "v570"
              ]
             ]
            ],
            [
             "mul",
             "v212",
             "v213"
            ]
           ]
          ],
          [
           "mul",
           "v214",
           "v215"
          ]
         ]
        ],
        [
         "mul",
         "v216",
         "v217"
        ]
       ]
      ],
      [
       "mul",
       "v218",
       "v219"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v207",
      [
       "add",
       [
        "mul",
        "v208",
        [
         "add",
         [
          "mul",
          "v209",
          [
           "add",
           [
            "mul",
            "v210",
            [
             "add",
             [
              "mul",
              "v565",
              [
               "add",
               [
                "mul",
                "v566",
                "v567"
               ],
               "v568"
              ]
             ],
             [
              "mul",
              "v569",
              "v570"
             ]
            ]
           ],
           [
            "mul",
            "v212",
            "v213"
           ]
          ]
         ],
         [
          "mul",
          "v214",
          "v215"
         ]
        ]
       ],
       [
        "mul",
        "v216",
        "v217"
       ]
      ]
     ],
     [
      "mul",
      "v218",
      "v219"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v565": [
        2,
        16,
        4
       ],
       "v566": [
        2,
        16,
        4
       ],
       "v567": [
        2,
        16,
        4
       ],
       "v568": [
        2,
        16,
        4
       ],
       "v569": [
        2,
        16,
        1
       ],
       "v570": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v565": [
        2,
        16,
        4
       ],
       "v566": [
        2,
        16,
        4
       ],
       "v567": [
        2,
        16,
        4
       ],
       "v568": [
        2,
        16,
        4
       ],
       "v569": [
        2,
        16,
        1
       ],
       "v570": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v565": [
        2,
        16,
        4
       ],
       "v566": [
        2,
        16,
        4
       ],
       "v567": [
        2,
        16,
        4
       ],
       "v568": [
        2,
        16,
        4
       ],
       "v569": [
        2,
        16,
        1
       ],
       "v570": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        1
       ],
       "v208": [
        2,
        16,
        1
       ],
       "v209": [
        2,
        16,
        1
       ],
       "v210": [
        2,
        16,
        1
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        16
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        16
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        16
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        16
       ],
       "v565": [
        2,
        16,
        1
       ],
       "v566": [
        2,
        16,
        1
       ],
       "v567": [
        2,
        16,
        16
       ],
       "v568": [
        2,
        16,
        16
       ],
       "v569": [
        2,
        16,
        1
       ],
       "v570": [
        2,
        1,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_21b9729f32": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', 'v5'), (mul 'v6', 'v7'))), (mul 'v8', 'v9'))), (mul 'v10', 'v11'))), (mul 'v12', 'v13'))), (mul 'v14', 'v15'))",
  "alpha_fold": "(synop2_35bbd03301 'v0', 'v1', 'v2', 'v3', (synop_bb8c54c201 'v4', 'v5', 'v6', 'v7'), 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15')",
  "embeds": [
   "synop3_4cdc06b88f"
  ],
  "embeds_shipped": [
   "synop2_35bbd03301",
   "synop_bb8c54c201"
  ],
  "equal_sites": 12,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_21b9729f32 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v548"
     },
     {
      "mvar": "v549"
     },
     {
      "mvar": "v550"
     },
     {
      "mvar": "v551"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_21b9729f32"
   },
   "missing_hooks": [],
   "name": "synop3_21b9729f32.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v207"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v208"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v209"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v210"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v548"
                       },
                       {
                        "mvar": "v549"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v550"
                       },
                       {
                        "mvar": "v551"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v212"
                   },
                   {
                    "mvar": "v213"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v214"
               },
               {
                "mvar": "v215"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v216"
           },
           {
            "mvar": "v217"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v218"
       },
       {
        "mvar": "v219"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "args": [
       {
        "mvar": "v548"
       },
       {
        "mvar": "v549"
       },
       {
        "mvar": "v550"
       },
       {
        "mvar": "v551"
       }
      ],
      "attrs": {},
      "op": "synop_bb8c54c201"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop2_35bbd03301"
   },
   "missing_hooks": [],
   "name": "prim:synop3_21b9729f32",
   "rhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v548"
     },
     {
      "mvar": "v549"
     },
     {
      "mvar": "v550"
     },
     {
      "mvar": "v551"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_21b9729f32"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v207",
     "v208",
     "v209",
     "v210",
     "v548",
     "v549",
     "v550",
     "v551",
     "v212",
     "v213",
     "v214",
     "v215",
     "v216",
     "v217",
     "v218",
     "v219"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v207",
       [
        "add",
        [
         "mul",
         "v208",
         [
          "add",
          [
           "mul",
           "v209",
           [
            "add",
            [
             "mul",
             "v210",
             [
              "add",
              [
               "mul",
               "v548",
               "v549"
              ],
              [
               "mul",
               "v550",
               "v551"
              ]
             ]
            ],
            [
             "mul",
             "v212",
             "v213"
            ]
           ]
          ],
          [
           "mul",
           "v214",
           "v215"
          ]
         ]
        ],
        [
         "mul",
         "v216",
         "v217"
        ]
       ]
      ],
      [
       "mul",
       "v218",
       "v219"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v207",
      [
       "add",
       [
        "mul",
        "v208",
        [
         "add",
         [
          "mul",
          "v209",
          [
           "add",
           [
            "mul",
            "v210",
            [
             "add",
             [
              "mul",
              "v548",
              "v549"
             ],
             [
              "mul",
              "v550",
              "v551"
             ]
            ]
           ],
           [
            "mul",
            "v212",
            "v213"
           ]
          ]
         ],
         [
          "mul",
          "v214",
          "v215"
         ]
        ]
       ],
       [
        "mul",
        "v216",
        "v217"
       ]
      ]
     ],
     [
      "mul",
      "v218",
      "v219"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v548": [
        2,
        16,
        4
       ],
       "v549": [
        2,
        16,
        4
       ],
       "v550": [
        2,
        16,
        1
       ],
       "v551": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v548": [
        2,
        16,
        4
       ],
       "v549": [
        2,
        16,
        4
       ],
       "v550": [
        2,
        16,
        1
       ],
       "v551": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v548": [
        2,
        16,
        4
       ],
       "v549": [
        2,
        16,
        4
       ],
       "v550": [
        2,
        16,
        1
       ],
       "v551": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v548": [
        2,
        16,
        4
       ],
       "v549": [
        2,
        16,
        4
       ],
       "v550": [
        2,
        16,
        1
       ],
       "v551": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 12
 },
 "synop3_24c5153e5e": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add 'v4', (mul 'v5', 'v6'))), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))), (mul 'v13', 'v14'))",
  "alpha_fold": "(synop2_35bbd03301 'v0', 'v1', 'v2', 'v3', (add 'v4', (mul 'v5', 'v6')), 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14')",
  "embeds": [],
  "embeds_shipped": [
   "synop2_35bbd03301"
  ],
  "equal_sites": 12,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_24c5153e5e unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v451"
     },
     {
      "mvar": "v1722"
     },
     {
      "mvar": "v1723"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_24c5153e5e"
   },
   "missing_hooks": [],
   "name": "synop3_24c5153e5e.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v207"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v208"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v209"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v210"
                   },
                   {
                    "args": [
                     {
                      "mvar": "v451"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v1722"
                       },
                       {
                        "mvar": "v1723"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v212"
                   },
                   {
                    "mvar": "v213"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v214"
               },
               {
                "mvar": "v215"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v216"
           },
           {
            "mvar": "v217"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v218"
       },
       {
        "mvar": "v219"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "args": [
       {
        "mvar": "v451"
       },
       {
        "args": [
         {
          "mvar": "v1722"
         },
         {
          "mvar": "v1723"
         }
        ],
        "attrs": {},
        "op": "mul"
       }
      ],
      "attrs": {},
      "op": "add"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop2_35bbd03301"
   },
   "missing_hooks": [],
   "name": "prim:synop3_24c5153e5e",
   "rhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v451"
     },
     {
      "mvar": "v1722"
     },
     {
      "mvar": "v1723"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_24c5153e5e"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v207",
     "v208",
     "v209",
     "v210",
     "v451",
     "v1722",
     "v1723",
     "v212",
     "v213",
     "v214",
     "v215",
     "v216",
     "v217",
     "v218",
     "v219"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v207",
       [
        "add",
        [
         "mul",
         "v208",
         [
          "add",
          [
           "mul",
           "v209",
           [
            "add",
            [
             "mul",
             "v210",
             [
              "add",
              "v451",
              [
               "mul",
               "v1722",
               "v1723"
              ]
             ]
            ],
            [
             "mul",
             "v212",
             "v213"
            ]
           ]
          ],
          [
           "mul",
           "v214",
           "v215"
          ]
         ]
        ],
        [
         "mul",
         "v216",
         "v217"
        ]
       ]
      ],
      [
       "mul",
       "v218",
       "v219"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v207",
      [
       "add",
       [
        "mul",
        "v208",
        [
         "add",
         [
          "mul",
          "v209",
          [
           "add",
           [
            "mul",
            "v210",
            [
             "add",
             "v451",
             [
              "mul",
              "v1722",
              "v1723"
             ]
            ]
           ],
           [
            "mul",
            "v212",
            "v213"
           ]
          ]
         ],
         [
          "mul",
          "v214",
          "v215"
         ]
        ]
       ],
       [
        "mul",
        "v216",
        "v217"
       ]
      ]
     ],
     [
      "mul",
      "v218",
      "v219"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v1722": [
        2,
        16,
        1
       ],
       "v1723": [
        2,
        1,
        4
       ],
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v451": [
        2,
        16,
        4
       ]
      },
      {
       "v1722": [
        2,
        16,
        1
       ],
       "v1723": [
        2,
        1,
        4
       ],
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v451": [
        2,
        16,
        4
       ]
      },
      {
       "v1722": [
        2,
        16,
        1
       ],
       "v1723": [
        2,
        1,
        4
       ],
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v451": [
        2,
        16,
        4
       ]
      },
      {
       "v1722": [
        2,
        16,
        1
       ],
       "v1723": [
        2,
        1,
        4
       ],
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v451": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 12
 },
 "synop3_2fcad5e803": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', 'v5'), (mul 'v6', 'v7'))), (mul 'v8', 'v9'))), (mul 'v10', 'v11'))), (mul 'v12', 'v13'))), (mul 'v14', 'v15'))",
  "alpha_fold": "(synop2_35bbd03301 'v0', 'v1', 'v2', 'v3', (add (mul 'v4', 'v5'), (mul 'v6', 'v7')), 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15')",
  "embeds": [
   "synop3_4cdc06b88f"
  ],
  "embeds_shipped": [
   "synop2_35bbd03301"
  ],
  "equal_sites": 12,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_2fcad5e803 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v1681"
     },
     {
      "mvar": "v1682"
     },
     {
      "mvar": "v1752"
     },
     {
      "mvar": "v1753"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_2fcad5e803"
   },
   "missing_hooks": [],
   "name": "synop3_2fcad5e803.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v207"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v208"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v209"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v210"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v1681"
                       },
                       {
                        "mvar": "v1682"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v1752"
                       },
                       {
                        "mvar": "v1753"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v212"
                   },
                   {
                    "mvar": "v213"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v214"
               },
               {
                "mvar": "v215"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v216"
           },
           {
            "mvar": "v217"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v218"
       },
       {
        "mvar": "v219"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v1681"
         },
         {
          "mvar": "v1682"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "args": [
         {
          "mvar": "v1752"
         },
         {
          "mvar": "v1753"
         }
        ],
        "attrs": {},
        "op": "mul"
       }
      ],
      "attrs": {},
      "op": "add"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop2_35bbd03301"
   },
   "missing_hooks": [],
   "name": "prim:synop3_2fcad5e803",
   "rhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v1681"
     },
     {
      "mvar": "v1682"
     },
     {
      "mvar": "v1752"
     },
     {
      "mvar": "v1753"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_2fcad5e803"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v207",
     "v208",
     "v209",
     "v210",
     "v1681",
     "v1682",
     "v1752",
     "v1753",
     "v212",
     "v213",
     "v214",
     "v215",
     "v216",
     "v217",
     "v218",
     "v219"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v207",
       [
        "add",
        [
         "mul",
         "v208",
         [
          "add",
          [
           "mul",
           "v209",
           [
            "add",
            [
             "mul",
             "v210",
             [
              "add",
              [
               "mul",
               "v1681",
               "v1682"
              ],
              [
               "mul",
               "v1752",
               "v1753"
              ]
             ]
            ],
            [
             "mul",
             "v212",
             "v213"
            ]
           ]
          ],
          [
           "mul",
           "v214",
           "v215"
          ]
         ]
        ],
        [
         "mul",
         "v216",
         "v217"
        ]
       ]
      ],
      [
       "mul",
       "v218",
       "v219"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v207",
      [
       "add",
       [
        "mul",
        "v208",
        [
         "add",
         [
          "mul",
          "v209",
          [
           "add",
           [
            "mul",
            "v210",
            [
             "add",
             [
              "mul",
              "v1681",
              "v1682"
             ],
             [
              "mul",
              "v1752",
              "v1753"
             ]
            ]
           ],
           [
            "mul",
            "v212",
            "v213"
           ]
          ]
         ],
         [
          "mul",
          "v214",
          "v215"
         ]
        ]
       ],
       [
        "mul",
        "v216",
        "v217"
       ]
      ]
     ],
     [
      "mul",
      "v218",
      "v219"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v1681": [
        2,
        16,
        4
       ],
       "v1682": [
        2,
        16,
        4
       ],
       "v1752": [
        2,
        16,
        1
       ],
       "v1753": [
        2,
        1,
        4
       ],
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ]
      },
      {
       "v1681": [
        2,
        16,
        4
       ],
       "v1682": [
        2,
        16,
        4
       ],
       "v1752": [
        2,
        16,
        1
       ],
       "v1753": [
        2,
        1,
        4
       ],
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ]
      },
      {
       "v1681": [
        2,
        16,
        4
       ],
       "v1682": [
        2,
        16,
        4
       ],
       "v1752": [
        2,
        16,
        1
       ],
       "v1753": [
        2,
        1,
        4
       ],
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ]
      },
      {
       "v1681": [
        2,
        16,
        4
       ],
       "v1682": [
        2,
        16,
        4
       ],
       "v1752": [
        2,
        16,
        1
       ],
       "v1753": [
        2,
        1,
        4
       ],
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 12
 },
 "synop3_304457cd8d": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', 'v5'), (mul 'v6', 'v7'))), (mul 'v8', 'v9'))), (mul 'v10', 'v11'))), (mul 'v12', 'v13'))), (mul 'v14', 'v15'))",
  "alpha_fold": "(synop2_d46808a970 'v0', 'v1', 'v2', 'v3', (synop_bb8c54c201 'v4', 'v5', 'v6', 'v7'), 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15')",
  "embeds": [
   "synop3_4cdc06b88f"
  ],
  "embeds_shipped": [
   "synop2_d46808a970",
   "synop_bb8c54c201"
  ],
  "equal_sites": 12,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_304457cd8d unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v696"
     },
     {
      "mvar": "v697"
     },
     {
      "mvar": "v698"
     },
     {
      "mvar": "v699"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_304457cd8d"
   },
   "missing_hooks": [],
   "name": "synop3_304457cd8d.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v220"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v221"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v222"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v223"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v696"
                       },
                       {
                        "mvar": "v697"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v698"
                       },
                       {
                        "mvar": "v699"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v225"
                   },
                   {
                    "mvar": "v226"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v227"
               },
               {
                "mvar": "v228"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v229"
           },
           {
            "mvar": "v230"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v231"
       },
       {
        "mvar": "v232"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "args": [
       {
        "mvar": "v696"
       },
       {
        "mvar": "v697"
       },
       {
        "mvar": "v698"
       },
       {
        "mvar": "v699"
       }
      ],
      "attrs": {},
      "op": "synop_bb8c54c201"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop2_d46808a970"
   },
   "missing_hooks": [],
   "name": "prim:synop3_304457cd8d",
   "rhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v696"
     },
     {
      "mvar": "v697"
     },
     {
      "mvar": "v698"
     },
     {
      "mvar": "v699"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_304457cd8d"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v220",
     "v221",
     "v222",
     "v223",
     "v696",
     "v697",
     "v698",
     "v699",
     "v225",
     "v226",
     "v227",
     "v228",
     "v229",
     "v230",
     "v231",
     "v232"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v220",
       [
        "add",
        [
         "mul",
         "v221",
         [
          "add",
          [
           "mul",
           "v222",
           [
            "add",
            [
             "mul",
             "v223",
             [
              "add",
              [
               "mul",
               "v696",
               "v697"
              ],
              [
               "mul",
               "v698",
               "v699"
              ]
             ]
            ],
            [
             "mul",
             "v225",
             "v226"
            ]
           ]
          ],
          [
           "mul",
           "v227",
           "v228"
          ]
         ]
        ],
        [
         "mul",
         "v229",
         "v230"
        ]
       ]
      ],
      [
       "mul",
       "v231",
       "v232"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v220",
      [
       "add",
       [
        "mul",
        "v221",
        [
         "add",
         [
          "mul",
          "v222",
          [
           "add",
           [
            "mul",
            "v223",
            [
             "add",
             [
              "mul",
              "v696",
              "v697"
             ],
             [
              "mul",
              "v698",
              "v699"
             ]
            ]
           ],
           [
            "mul",
            "v225",
            "v226"
           ]
          ]
         ],
         [
          "mul",
          "v227",
          "v228"
         ]
        ]
       ],
       [
        "mul",
        "v229",
        "v230"
       ]
      ]
     ],
     [
      "mul",
      "v231",
      "v232"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v696": [
        2,
        16,
        4
       ],
       "v697": [
        2,
        16,
        4
       ],
       "v698": [
        2,
        16,
        1
       ],
       "v699": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v696": [
        2,
        16,
        4
       ],
       "v697": [
        2,
        16,
        4
       ],
       "v698": [
        2,
        16,
        1
       ],
       "v699": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v696": [
        2,
        16,
        4
       ],
       "v697": [
        2,
        16,
        4
       ],
       "v698": [
        2,
        16,
        1
       ],
       "v699": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v696": [
        2,
        16,
        4
       ],
       "v697": [
        2,
        16,
        4
       ],
       "v698": [
        2,
        16,
        1
       ],
       "v699": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 12
 },
 "synop3_39dbcda988": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add 'v4', (mul 'v5', 'v6'))), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))), (mul 'v13', 'v14'))",
  "alpha_fold": "(synop2_d46808a970 'v0', 'v1', 'v2', 'v3', (add 'v4', (mul 'v5', 'v6')), 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14')",
  "embeds": [],
  "embeds_shipped": [
   "synop2_d46808a970"
  ],
  "equal_sites": 12,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_39dbcda988 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v599"
     },
     {
      "mvar": "v2224"
     },
     {
      "mvar": "v2225"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_39dbcda988"
   },
   "missing_hooks": [],
   "name": "synop3_39dbcda988.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v220"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v221"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v222"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v223"
                   },
                   {
                    "args": [
                     {
                      "mvar": "v599"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v2224"
                       },
                       {
                        "mvar": "v2225"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v225"
                   },
                   {
                    "mvar": "v226"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v227"
               },
               {
                "mvar": "v228"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v229"
           },
           {
            "mvar": "v230"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v231"
       },
       {
        "mvar": "v232"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "args": [
       {
        "mvar": "v599"
       },
       {
        "args": [
         {
          "mvar": "v2224"
         },
         {
          "mvar": "v2225"
         }
        ],
        "attrs": {},
        "op": "mul"
       }
      ],
      "attrs": {},
      "op": "add"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop2_d46808a970"
   },
   "missing_hooks": [],
   "name": "prim:synop3_39dbcda988",
   "rhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v599"
     },
     {
      "mvar": "v2224"
     },
     {
      "mvar": "v2225"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_39dbcda988"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v220",
     "v221",
     "v222",
     "v223",
     "v599",
     "v2224",
     "v2225",
     "v225",
     "v226",
     "v227",
     "v228",
     "v229",
     "v230",
     "v231",
     "v232"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v220",
       [
        "add",
        [
         "mul",
         "v221",
         [
          "add",
          [
           "mul",
           "v222",
           [
            "add",
            [
             "mul",
             "v223",
             [
              "add",
              "v599",
              [
               "mul",
               "v2224",
               "v2225"
              ]
             ]
            ],
            [
             "mul",
             "v225",
             "v226"
            ]
           ]
          ],
          [
           "mul",
           "v227",
           "v228"
          ]
         ]
        ],
        [
         "mul",
         "v229",
         "v230"
        ]
       ]
      ],
      [
       "mul",
       "v231",
       "v232"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v220",
      [
       "add",
       [
        "mul",
        "v221",
        [
         "add",
         [
          "mul",
          "v222",
          [
           "add",
           [
            "mul",
            "v223",
            [
             "add",
             "v599",
             [
              "mul",
              "v2224",
              "v2225"
             ]
            ]
           ],
           [
            "mul",
            "v225",
            "v226"
           ]
          ]
         ],
         [
          "mul",
          "v227",
          "v228"
         ]
        ]
       ],
       [
        "mul",
        "v229",
        "v230"
       ]
      ]
     ],
     [
      "mul",
      "v231",
      "v232"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v2224": [
        2,
        16,
        1
       ],
       "v2225": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v599": [
        2,
        16,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v2224": [
        2,
        16,
        1
       ],
       "v2225": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v599": [
        2,
        16,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v2224": [
        2,
        16,
        1
       ],
       "v2225": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v599": [
        2,
        16,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v2224": [
        2,
        16,
        1
       ],
       "v2225": [
        2,
        1,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v599": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 12
 },
 "synop3_426efb3fb1": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))), (mul 'v13', 'v14'))), (mul 'v15', 'v16'))), (mul 'v17', 'v18'))",
  "alpha_fold": "(synop2_d46808a970 'v0', 'v1', 'v2', 'v3', (synop_2b8cfd1c90 'v4', (mul 'v5', 'v6'), 'v7', 'v8', 'v9', 'v10'), 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18')",
  "embeds": [
   "synop3_304457cd8d",
   "synop3_4cdc06b88f"
  ],
  "embeds_shipped": [
   "synop2_d46808a970",
   "synop_2b8cfd1c90"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_426efb3fb1 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v633"
     },
     {
      "mvar": "v3272"
     },
     {
      "mvar": "v3273"
     },
     {
      "mvar": "v635"
     },
     {
      "mvar": "v636"
     },
     {
      "mvar": "v637"
     },
     {
      "mvar": "v638"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_426efb3fb1"
   },
   "missing_hooks": [],
   "name": "synop3_426efb3fb1.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v220"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v221"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v222"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v223"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v633"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v3272"
                           },
                           {
                            "mvar": "v3273"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v635"
                           },
                           {
                            "mvar": "v636"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v637"
                       },
                       {
                        "mvar": "v638"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v225"
                   },
                   {
                    "mvar": "v226"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v227"
               },
               {
                "mvar": "v228"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v229"
           },
           {
            "mvar": "v230"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v231"
       },
       {
        "mvar": "v232"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "args": [
       {
        "mvar": "v633"
       },
       {
        "args": [
         {
          "mvar": "v3272"
         },
         {
          "mvar": "v3273"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "mvar": "v635"
       },
       {
        "mvar": "v636"
       },
       {
        "mvar": "v637"
       },
       {
        "mvar": "v638"
       }
      ],
      "attrs": {},
      "op": "synop_2b8cfd1c90"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop2_d46808a970"
   },
   "missing_hooks": [],
   "name": "prim:synop3_426efb3fb1",
   "rhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v633"
     },
     {
      "mvar": "v3272"
     },
     {
      "mvar": "v3273"
     },
     {
      "mvar": "v635"
     },
     {
      "mvar": "v636"
     },
     {
      "mvar": "v637"
     },
     {
      "mvar": "v638"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_426efb3fb1"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v220",
     "v221",
     "v222",
     "v223",
     "v633",
     "v3272",
     "v3273",
     "v635",
     "v636",
     "v637",
     "v638",
     "v225",
     "v226",
     "v227",
     "v228",
     "v229",
     "v230",
     "v231",
     "v232"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v220",
       [
        "add",
        [
         "mul",
         "v221",
         [
          "add",
          [
           "mul",
           "v222",
           [
            "add",
            [
             "mul",
             "v223",
             [
              "add",
              [
               "mul",
               "v633",
               [
                "add",
                [
                 "mul",
                 "v3272",
                 "v3273"
                ],
                [
                 "mul",
                 "v635",
                 "v636"
                ]
               ]
              ],
              [
               "mul",
               "v637",
               "v638"
              ]
             ]
            ],
            [
             "mul",
             "v225",
             "v226"
            ]
           ]
          ],
          [
           "mul",
           "v227",
           "v228"
          ]
         ]
        ],
        [
         "mul",
         "v229",
         "v230"
        ]
       ]
      ],
      [
       "mul",
       "v231",
       "v232"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v220",
      [
       "add",
       [
        "mul",
        "v221",
        [
         "add",
         [
          "mul",
          "v222",
          [
           "add",
           [
            "mul",
            "v223",
            [
             "add",
             [
              "mul",
              "v633",
              [
               "add",
               [
                "mul",
                "v3272",
                "v3273"
               ],
               [
                "mul",
                "v635",
                "v636"
               ]
              ]
             ],
             [
              "mul",
              "v637",
              "v638"
             ]
            ]
           ],
           [
            "mul",
            "v225",
            "v226"
           ]
          ]
         ],
         [
          "mul",
          "v227",
          "v228"
         ]
        ]
       ],
       [
        "mul",
        "v229",
        "v230"
       ]
      ]
     ],
     [
      "mul",
      "v231",
      "v232"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v3272": [
        2,
        16,
        4
       ],
       "v3273": [
        2,
        16,
        4
       ],
       "v633": [
        2,
        16,
        4
       ],
       "v635": [
        2,
        16,
        1
       ],
       "v636": [
        2,
        1,
        4
       ],
       "v637": [
        2,
        16,
        1
       ],
       "v638": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v3272": [
        2,
        16,
        4
       ],
       "v3273": [
        2,
        16,
        4
       ],
       "v633": [
        2,
        16,
        4
       ],
       "v635": [
        2,
        16,
        1
       ],
       "v636": [
        2,
        1,
        4
       ],
       "v637": [
        2,
        16,
        1
       ],
       "v638": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v3272": [
        2,
        16,
        4
       ],
       "v3273": [
        2,
        16,
        4
       ],
       "v633": [
        2,
        16,
        4
       ],
       "v635": [
        2,
        16,
        1
       ],
       "v636": [
        2,
        1,
        4
       ],
       "v637": [
        2,
        16,
        1
       ],
       "v638": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        1
       ],
       "v221": [
        2,
        16,
        1
       ],
       "v222": [
        2,
        16,
        1
       ],
       "v223": [
        2,
        16,
        1
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        16
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        16
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        16
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        16
       ],
       "v3272": [
        2,
        16,
        1
       ],
       "v3273": [
        2,
        16,
        16
       ],
       "v633": [
        2,
        16,
        1
       ],
       "v635": [
        2,
        16,
        1
       ],
       "v636": [
        2,
        1,
        16
       ],
       "v637": [
        2,
        16,
        1
       ],
       "v638": [
        2,
        1,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_42bb04032d": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))), (mul 'v13', 'v14'))), (mul 'v15', 'v16'))), (mul 'v17', 'v18'))",
  "alpha_fold": "(synop2_35bbd03301 'v0', 'v1', 'v2', 'v3', (synop_5e58f26875 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'), 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18')",
  "embeds": [
   "synop3_304457cd8d",
   "synop3_4cdc06b88f"
  ],
  "embeds_shipped": [
   "synop2_35bbd03301",
   "synop_5e58f26875"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_42bb04032d unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v523"
     },
     {
      "mvar": "v524"
     },
     {
      "mvar": "v525"
     },
     {
      "mvar": "v526"
     },
     {
      "mvar": "v527"
     },
     {
      "mvar": "v528"
     },
     {
      "mvar": "v529"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_42bb04032d"
   },
   "missing_hooks": [],
   "name": "synop3_42bb04032d.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v207"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v208"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v209"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v210"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v523"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v524"
                           },
                           {
                            "mvar": "v525"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v526"
                           },
                           {
                            "mvar": "v527"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v528"
                       },
                       {
                        "mvar": "v529"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v212"
                   },
                   {
                    "mvar": "v213"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v214"
               },
               {
                "mvar": "v215"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v216"
           },
           {
            "mvar": "v217"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v218"
       },
       {
        "mvar": "v219"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "args": [
       {
        "mvar": "v523"
       },
       {
        "mvar": "v524"
       },
       {
        "mvar": "v525"
       },
       {
        "mvar": "v526"
       },
       {
        "mvar": "v527"
       },
       {
        "mvar": "v528"
       },
       {
        "mvar": "v529"
       }
      ],
      "attrs": {},
      "op": "synop_5e58f26875"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop2_35bbd03301"
   },
   "missing_hooks": [],
   "name": "prim:synop3_42bb04032d",
   "rhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v523"
     },
     {
      "mvar": "v524"
     },
     {
      "mvar": "v525"
     },
     {
      "mvar": "v526"
     },
     {
      "mvar": "v527"
     },
     {
      "mvar": "v528"
     },
     {
      "mvar": "v529"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_42bb04032d"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v207",
     "v208",
     "v209",
     "v210",
     "v523",
     "v524",
     "v525",
     "v526",
     "v527",
     "v528",
     "v529",
     "v212",
     "v213",
     "v214",
     "v215",
     "v216",
     "v217",
     "v218",
     "v219"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v207",
       [
        "add",
        [
         "mul",
         "v208",
         [
          "add",
          [
           "mul",
           "v209",
           [
            "add",
            [
             "mul",
             "v210",
             [
              "add",
              [
               "mul",
               "v523",
               [
                "add",
                [
                 "mul",
                 "v524",
                 "v525"
                ],
                [
                 "mul",
                 "v526",
                 "v527"
                ]
               ]
              ],
              [
               "mul",
               "v528",
               "v529"
              ]
             ]
            ],
            [
             "mul",
             "v212",
             "v213"
            ]
           ]
          ],
          [
           "mul",
           "v214",
           "v215"
          ]
         ]
        ],
        [
         "mul",
         "v216",
         "v217"
        ]
       ]
      ],
      [
       "mul",
       "v218",
       "v219"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v207",
      [
       "add",
       [
        "mul",
        "v208",
        [
         "add",
         [
          "mul",
          "v209",
          [
           "add",
           [
            "mul",
            "v210",
            [
             "add",
             [
              "mul",
              "v523",
              [
               "add",
               [
                "mul",
                "v524",
                "v525"
               ],
               [
                "mul",
                "v526",
                "v527"
               ]
              ]
             ],
             [
              "mul",
              "v528",
              "v529"
             ]
            ]
           ],
           [
            "mul",
            "v212",
            "v213"
           ]
          ]
         ],
         [
          "mul",
          "v214",
          "v215"
         ]
        ]
       ],
       [
        "mul",
        "v216",
        "v217"
       ]
      ]
     ],
     [
      "mul",
      "v218",
      "v219"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v523": [
        2,
        16,
        4
       ],
       "v524": [
        2,
        16,
        4
       ],
       "v525": [
        2,
        16,
        4
       ],
       "v526": [
        2,
        16,
        1
       ],
       "v527": [
        2,
        1,
        4
       ],
       "v528": [
        2,
        16,
        1
       ],
       "v529": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v523": [
        2,
        16,
        4
       ],
       "v524": [
        2,
        16,
        4
       ],
       "v525": [
        2,
        16,
        4
       ],
       "v526": [
        2,
        16,
        1
       ],
       "v527": [
        2,
        1,
        4
       ],
       "v528": [
        2,
        16,
        1
       ],
       "v529": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v523": [
        2,
        16,
        4
       ],
       "v524": [
        2,
        16,
        4
       ],
       "v525": [
        2,
        16,
        4
       ],
       "v526": [
        2,
        16,
        1
       ],
       "v527": [
        2,
        1,
        4
       ],
       "v528": [
        2,
        16,
        1
       ],
       "v529": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        1
       ],
       "v208": [
        2,
        16,
        1
       ],
       "v209": [
        2,
        16,
        1
       ],
       "v210": [
        2,
        16,
        1
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        16
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        16
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        16
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        16
       ],
       "v523": [
        2,
        16,
        1
       ],
       "v524": [
        2,
        16,
        1
       ],
       "v525": [
        2,
        16,
        16
       ],
       "v526": [
        2,
        16,
        1
       ],
       "v527": [
        2,
        1,
        16
       ],
       "v528": [
        2,
        16,
        1
       ],
       "v529": [
        2,
        1,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_42f015e162": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))), (mul 'v13', 'v14'))), (mul 'v15', 'v16'))), (mul 'v17', 'v18'))",
  "alpha_fold": "(synop2_d46808a970 'v0', 'v1', 'v2', 'v3', (add (mul 'v4', (synop_bb8c54c201 'v5', 'v6', 'v7', 'v8')), (mul 'v9', 'v10')), 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18')",
  "embeds": [
   "synop3_304457cd8d",
   "synop3_4cdc06b88f"
  ],
  "embeds_shipped": [
   "synop2_d46808a970",
   "synop_bb8c54c201"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_42f015e162 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v2183"
     },
     {
      "mvar": "v2453"
     },
     {
      "mvar": "v2454"
     },
     {
      "mvar": "v2455"
     },
     {
      "mvar": "v2456"
     },
     {
      "mvar": "v2254"
     },
     {
      "mvar": "v2255"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_42f015e162"
   },
   "missing_hooks": [],
   "name": "synop3_42f015e162.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v220"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v221"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v222"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v223"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v2183"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v2453"
                           },
                           {
                            "mvar": "v2454"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v2455"
                           },
                           {
                            "mvar": "v2456"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v2254"
                       },
                       {
                        "mvar": "v2255"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v225"
                   },
                   {
                    "mvar": "v226"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v227"
               },
               {
                "mvar": "v228"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v229"
           },
           {
            "mvar": "v230"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v231"
       },
       {
        "mvar": "v232"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v2183"
         },
         {
          "args": [
           {
            "mvar": "v2453"
           },
           {
            "mvar": "v2454"
           },
           {
            "mvar": "v2455"
           },
           {
            "mvar": "v2456"
           }
          ],
          "attrs": {},
          "op": "synop_bb8c54c201"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "args": [
         {
          "mvar": "v2254"
         },
         {
          "mvar": "v2255"
         }
        ],
        "attrs": {},
        "op": "mul"
       }
      ],
      "attrs": {},
      "op": "add"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop2_d46808a970"
   },
   "missing_hooks": [],
   "name": "prim:synop3_42f015e162",
   "rhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v2183"
     },
     {
      "mvar": "v2453"
     },
     {
      "mvar": "v2454"
     },
     {
      "mvar": "v2455"
     },
     {
      "mvar": "v2456"
     },
     {
      "mvar": "v2254"
     },
     {
      "mvar": "v2255"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_42f015e162"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v220",
     "v221",
     "v222",
     "v223",
     "v2183",
     "v2453",
     "v2454",
     "v2455",
     "v2456",
     "v2254",
     "v2255",
     "v225",
     "v226",
     "v227",
     "v228",
     "v229",
     "v230",
     "v231",
     "v232"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v220",
       [
        "add",
        [
         "mul",
         "v221",
         [
          "add",
          [
           "mul",
           "v222",
           [
            "add",
            [
             "mul",
             "v223",
             [
              "add",
              [
               "mul",
               "v2183",
               [
                "add",
                [
                 "mul",
                 "v2453",
                 "v2454"
                ],
                [
                 "mul",
                 "v2455",
                 "v2456"
                ]
               ]
              ],
              [
               "mul",
               "v2254",
               "v2255"
              ]
             ]
            ],
            [
             "mul",
             "v225",
             "v226"
            ]
           ]
          ],
          [
           "mul",
           "v227",
           "v228"
          ]
         ]
        ],
        [
         "mul",
         "v229",
         "v230"
        ]
       ]
      ],
      [
       "mul",
       "v231",
       "v232"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v220",
      [
       "add",
       [
        "mul",
        "v221",
        [
         "add",
         [
          "mul",
          "v222",
          [
           "add",
           [
            "mul",
            "v223",
            [
             "add",
             [
              "mul",
              "v2183",
              [
               "add",
               [
                "mul",
                "v2453",
                "v2454"
               ],
               [
                "mul",
                "v2455",
                "v2456"
               ]
              ]
             ],
             [
              "mul",
              "v2254",
              "v2255"
             ]
            ]
           ],
           [
            "mul",
            "v225",
            "v226"
           ]
          ]
         ],
         [
          "mul",
          "v227",
          "v228"
         ]
        ]
       ],
       [
        "mul",
        "v229",
        "v230"
       ]
      ]
     ],
     [
      "mul",
      "v231",
      "v232"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v2183": [
        2,
        16,
        4
       ],
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v2254": [
        2,
        16,
        1
       ],
       "v2255": [
        2,
        1,
        4
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v2453": [
        2,
        16,
        4
       ],
       "v2454": [
        2,
        16,
        4
       ],
       "v2455": [
        2,
        16,
        1
       ],
       "v2456": [
        2,
        1,
        4
       ]
      },
      {
       "v2183": [
        2,
        16,
        4
       ],
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v2254": [
        2,
        16,
        1
       ],
       "v2255": [
        2,
        1,
        4
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v2453": [
        2,
        16,
        4
       ],
       "v2454": [
        2,
        16,
        4
       ],
       "v2455": [
        2,
        16,
        1
       ],
       "v2456": [
        2,
        1,
        4
       ]
      },
      {
       "v2183": [
        2,
        16,
        4
       ],
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v2254": [
        2,
        16,
        1
       ],
       "v2255": [
        2,
        1,
        4
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v2453": [
        2,
        16,
        4
       ],
       "v2454": [
        2,
        16,
        4
       ],
       "v2455": [
        2,
        16,
        1
       ],
       "v2456": [
        2,
        1,
        4
       ]
      },
      {
       "v2183": [
        2,
        16,
        1
       ],
       "v220": [
        2,
        16,
        1
       ],
       "v221": [
        2,
        16,
        1
       ],
       "v222": [
        2,
        16,
        1
       ],
       "v223": [
        2,
        16,
        1
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v2254": [
        2,
        16,
        1
       ],
       "v2255": [
        2,
        1,
        16
       ],
       "v226": [
        2,
        1,
        16
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        16
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        16
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        16
       ],
       "v2453": [
        2,
        16,
        1
       ],
       "v2454": [
        2,
        16,
        16
       ],
       "v2455": [
        2,
        16,
        1
       ],
       "v2456": [
        2,
        1,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_490109134e": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))), (mul 'v13', 'v14'))), (mul 'v15', 'v16'))), (mul 'v17', 'v18'))",
  "alpha_fold": "(synop2_d46808a970 'v0', 'v1', 'v2', 'v3', (synop_5e58f26875 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'), 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18')",
  "embeds": [
   "synop3_304457cd8d",
   "synop3_4cdc06b88f"
  ],
  "embeds_shipped": [
   "synop2_d46808a970",
   "synop_5e58f26875"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_490109134e unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v671"
     },
     {
      "mvar": "v672"
     },
     {
      "mvar": "v673"
     },
     {
      "mvar": "v674"
     },
     {
      "mvar": "v675"
     },
     {
      "mvar": "v676"
     },
     {
      "mvar": "v677"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_490109134e"
   },
   "missing_hooks": [],
   "name": "synop3_490109134e.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v220"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v221"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v222"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v223"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v671"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v672"
                           },
                           {
                            "mvar": "v673"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v674"
                           },
                           {
                            "mvar": "v675"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v676"
                       },
                       {
                        "mvar": "v677"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v225"
                   },
                   {
                    "mvar": "v226"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v227"
               },
               {
                "mvar": "v228"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v229"
           },
           {
            "mvar": "v230"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v231"
       },
       {
        "mvar": "v232"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "args": [
       {
        "mvar": "v671"
       },
       {
        "mvar": "v672"
       },
       {
        "mvar": "v673"
       },
       {
        "mvar": "v674"
       },
       {
        "mvar": "v675"
       },
       {
        "mvar": "v676"
       },
       {
        "mvar": "v677"
       }
      ],
      "attrs": {},
      "op": "synop_5e58f26875"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop2_d46808a970"
   },
   "missing_hooks": [],
   "name": "prim:synop3_490109134e",
   "rhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v671"
     },
     {
      "mvar": "v672"
     },
     {
      "mvar": "v673"
     },
     {
      "mvar": "v674"
     },
     {
      "mvar": "v675"
     },
     {
      "mvar": "v676"
     },
     {
      "mvar": "v677"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_490109134e"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v220",
     "v221",
     "v222",
     "v223",
     "v671",
     "v672",
     "v673",
     "v674",
     "v675",
     "v676",
     "v677",
     "v225",
     "v226",
     "v227",
     "v228",
     "v229",
     "v230",
     "v231",
     "v232"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v220",
       [
        "add",
        [
         "mul",
         "v221",
         [
          "add",
          [
           "mul",
           "v222",
           [
            "add",
            [
             "mul",
             "v223",
             [
              "add",
              [
               "mul",
               "v671",
               [
                "add",
                [
                 "mul",
                 "v672",
                 "v673"
                ],
                [
                 "mul",
                 "v674",
                 "v675"
                ]
               ]
              ],
              [
               "mul",
               "v676",
               "v677"
              ]
             ]
            ],
            [
             "mul",
             "v225",
             "v226"
            ]
           ]
          ],
          [
           "mul",
           "v227",
           "v228"
          ]
         ]
        ],
        [
         "mul",
         "v229",
         "v230"
        ]
       ]
      ],
      [
       "mul",
       "v231",
       "v232"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v220",
      [
       "add",
       [
        "mul",
        "v221",
        [
         "add",
         [
          "mul",
          "v222",
          [
           "add",
           [
            "mul",
            "v223",
            [
             "add",
             [
              "mul",
              "v671",
              [
               "add",
               [
                "mul",
                "v672",
                "v673"
               ],
               [
                "mul",
                "v674",
                "v675"
               ]
              ]
             ],
             [
              "mul",
              "v676",
              "v677"
             ]
            ]
           ],
           [
            "mul",
            "v225",
            "v226"
           ]
          ]
         ],
         [
          "mul",
          "v227",
          "v228"
         ]
        ]
       ],
       [
        "mul",
        "v229",
        "v230"
       ]
      ]
     ],
     [
      "mul",
      "v231",
      "v232"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v671": [
        2,
        16,
        4
       ],
       "v672": [
        2,
        16,
        4
       ],
       "v673": [
        2,
        16,
        4
       ],
       "v674": [
        2,
        16,
        1
       ],
       "v675": [
        2,
        1,
        4
       ],
       "v676": [
        2,
        16,
        1
       ],
       "v677": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v671": [
        2,
        16,
        4
       ],
       "v672": [
        2,
        16,
        4
       ],
       "v673": [
        2,
        16,
        4
       ],
       "v674": [
        2,
        16,
        1
       ],
       "v675": [
        2,
        1,
        4
       ],
       "v676": [
        2,
        16,
        1
       ],
       "v677": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v671": [
        2,
        16,
        4
       ],
       "v672": [
        2,
        16,
        4
       ],
       "v673": [
        2,
        16,
        4
       ],
       "v674": [
        2,
        16,
        1
       ],
       "v675": [
        2,
        1,
        4
       ],
       "v676": [
        2,
        16,
        1
       ],
       "v677": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        1
       ],
       "v221": [
        2,
        16,
        1
       ],
       "v222": [
        2,
        16,
        1
       ],
       "v223": [
        2,
        16,
        1
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        16
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        16
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        16
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        16
       ],
       "v671": [
        2,
        16,
        1
       ],
       "v672": [
        2,
        16,
        1
       ],
       "v673": [
        2,
        16,
        16
       ],
       "v674": [
        2,
        16,
        1
       ],
       "v675": [
        2,
        1,
        16
       ],
       "v676": [
        2,
        16,
        1
       ],
       "v677": [
        2,
        1,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_4a015e46ce": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))), (mul 'v13', 'v14'))), (mul 'v15', 'v16'))), (mul 'v17', 'v18'))",
  "alpha_fold": "(synop_5e58f26875 'v0', 'v1', (synop2_35bbd03301 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14'), 'v15', 'v16', 'v17', 'v18')",
  "embeds": [
   "synop3_304457cd8d",
   "synop3_4cdc06b88f"
  ],
  "embeds_shipped": [
   "synop2_35bbd03301",
   "synop_5e58f26875"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_4a015e46ce unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v115"
     },
     {
      "mvar": "v116"
     },
     {
      "mvar": "v4031"
     },
     {
      "mvar": "v4032"
     },
     {
      "mvar": "v4033"
     },
     {
      "mvar": "v4034"
     },
     {
      "mvar": "v4035"
     },
     {
      "mvar": "v4036"
     },
     {
      "mvar": "v4037"
     },
     {
      "mvar": "v4038"
     },
     {
      "mvar": "v4039"
     },
     {
      "mvar": "v4040"
     },
     {
      "mvar": "v4041"
     },
     {
      "mvar": "v4042"
     },
     {
      "mvar": "v4043"
     },
     {
      "mvar": "v118"
     },
     {
      "mvar": "v119"
     },
     {
      "mvar": "v120"
     },
     {
      "mvar": "v121"
     }
    ],
    "attrs": {},
    "op": "synop3_4a015e46ce"
   },
   "missing_hooks": [],
   "name": "synop3_4a015e46ce.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v115"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v116"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v4031"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v4032"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v4033"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v4034"
                           },
                           {
                            "mvar": "v4035"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v4036"
                           },
                           {
                            "mvar": "v4037"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v4038"
                       },
                       {
                        "mvar": "v4039"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v4040"
                   },
                   {
                    "mvar": "v4041"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v4042"
               },
               {
                "mvar": "v4043"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v118"
           },
           {
            "mvar": "v119"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v120"
       },
       {
        "mvar": "v121"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v115"
     },
     {
      "mvar": "v116"
     },
     {
      "args": [
       {
        "mvar": "v4031"
       },
       {
        "mvar": "v4032"
       },
       {
        "mvar": "v4033"
       },
       {
        "mvar": "v4034"
       },
       {
        "mvar": "v4035"
       },
       {
        "mvar": "v4036"
       },
       {
        "mvar": "v4037"
       },
       {
        "mvar": "v4038"
       },
       {
        "mvar": "v4039"
       },
       {
        "mvar": "v4040"
       },
       {
        "mvar": "v4041"
       },
       {
        "mvar": "v4042"
       },
       {
        "mvar": "v4043"
       }
      ],
      "attrs": {},
      "op": "synop2_35bbd03301"
     },
     {
      "mvar": "v118"
     },
     {
      "mvar": "v119"
     },
     {
      "mvar": "v120"
     },
     {
      "mvar": "v121"
     }
    ],
    "attrs": {},
    "op": "synop_5e58f26875"
   },
   "missing_hooks": [],
   "name": "prim:synop3_4a015e46ce",
   "rhs": {
    "args": [
     {
      "mvar": "v115"
     },
     {
      "mvar": "v116"
     },
     {
      "mvar": "v4031"
     },
     {
      "mvar": "v4032"
     },
     {
      "mvar": "v4033"
     },
     {
      "mvar": "v4034"
     },
     {
      "mvar": "v4035"
     },
     {
      "mvar": "v4036"
     },
     {
      "mvar": "v4037"
     },
     {
      "mvar": "v4038"
     },
     {
      "mvar": "v4039"
     },
     {
      "mvar": "v4040"
     },
     {
      "mvar": "v4041"
     },
     {
      "mvar": "v4042"
     },
     {
      "mvar": "v4043"
     },
     {
      "mvar": "v118"
     },
     {
      "mvar": "v119"
     },
     {
      "mvar": "v120"
     },
     {
      "mvar": "v121"
     }
    ],
    "attrs": {},
    "op": "synop3_4a015e46ce"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v115",
     "v116",
     "v4031",
     "v4032",
     "v4033",
     "v4034",
     "v4035",
     "v4036",
     "v4037",
     "v4038",
     "v4039",
     "v4040",
     "v4041",
     "v4042",
     "v4043",
     "v118",
     "v119",
     "v120",
     "v121"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v115",
       [
        "add",
        [
         "mul",
         "v116",
         [
          "add",
          [
           "mul",
           "v4031",
           [
            "add",
            [
             "mul",
             "v4032",
             [
              "add",
              [
               "mul",
               "v4033",
               [
                "add",
                [
                 "mul",
                 "v4034",
                 "v4035"
                ],
                [
                 "mul",
                 "v4036",
                 "v4037"
                ]
               ]
              ],
              [
               "mul",
               "v4038",
               "v4039"
              ]
             ]
            ],
            [
             "mul",
             "v4040",
             "v4041"
            ]
           ]
          ],
          [
           "mul",
           "v4042",
           "v4043"
          ]
         ]
        ],
        [
         "mul",
         "v118",
         "v119"
        ]
       ]
      ],
      [
       "mul",
       "v120",
       "v121"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v115",
      [
       "add",
       [
        "mul",
        "v116",
        [
         "add",
         [
          "mul",
          "v4031",
          [
           "add",
           [
            "mul",
            "v4032",
            [
             "add",
             [
              "mul",
              "v4033",
              [
               "add",
               [
                "mul",
                "v4034",
                "v4035"
               ],
               [
                "mul",
                "v4036",
                "v4037"
               ]
              ]
             ],
             [
              "mul",
              "v4038",
              "v4039"
             ]
            ]
           ],
           [
            "mul",
            "v4040",
            "v4041"
           ]
          ]
         ],
         [
          "mul",
          "v4042",
          "v4043"
         ]
        ]
       ],
       [
        "mul",
        "v118",
        "v119"
       ]
      ]
     ],
     [
      "mul",
      "v120",
      "v121"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v115": [
        2,
        16,
        4
       ],
       "v116": [
        2,
        16,
        4
       ],
       "v118": [
        2,
        16,
        1
       ],
       "v119": [
        2,
        1,
        4
       ],
       "v120": [
        2,
        16,
        1
       ],
       "v121": [
        2,
        1,
        4
       ],
       "v4031": [
        2,
        16,
        4
       ],
       "v4032": [
        2,
        16,
        4
       ],
       "v4033": [
        2,
        16,
        4
       ],
       "v4034": [
        2,
        16,
        4
       ],
       "v4035": [
        2,
        16,
        4
       ],
       "v4036": [
        2,
        16,
        1
       ],
       "v4037": [
        2,
        1,
        4
       ],
       "v4038": [
        2,
        16,
        1
       ],
       "v4039": [
        2,
        1,
        4
       ],
       "v4040": [
        2,
        16,
        1
       ],
       "v4041": [
        2,
        1,
        4
       ],
       "v4042": [
        2,
        16,
        1
       ],
       "v4043": [
        2,
        1,
        4
       ]
      },
      {
       "v115": [
        2,
        16,
        4
       ],
       "v116": [
        2,
        16,
        4
       ],
       "v118": [
        2,
        16,
        1
       ],
       "v119": [
        2,
        1,
        4
       ],
       "v120": [
        2,
        16,
        1
       ],
       "v121": [
        2,
        1,
        4
       ],
       "v4031": [
        2,
        16,
        4
       ],
       "v4032": [
        2,
        16,
        4
       ],
       "v4033": [
        2,
        16,
        4
       ],
       "v4034": [
        2,
        16,
        4
       ],
       "v4035": [
        2,
        16,
        4
       ],
       "v4036": [
        2,
        16,
        1
       ],
       "v4037": [
        2,
        1,
        4
       ],
       "v4038": [
        2,
        16,
        1
       ],
       "v4039": [
        2,
        1,
        4
       ],
       "v4040": [
        2,
        16,
        1
       ],
       "v4041": [
        2,
        1,
        4
       ],
       "v4042": [
        2,
        16,
        1
       ],
       "v4043": [
        2,
        1,
        4
       ]
      },
      {
       "v115": [
        2,
        16,
        4
       ],
       "v116": [
        2,
        16,
        4
       ],
       "v118": [
        2,
        16,
        1
       ],
       "v119": [
        2,
        1,
        4
       ],
       "v120": [
        2,
        16,
        1
       ],
       "v121": [
        2,
        1,
        4
       ],
       "v4031": [
        2,
        16,
        4
       ],
       "v4032": [
        2,
        16,
        4
       ],
       "v4033": [
        2,
        16,
        4
       ],
       "v4034": [
        2,
        16,
        4
       ],
       "v4035": [
        2,
        16,
        4
       ],
       "v4036": [
        2,
        16,
        1
       ],
       "v4037": [
        2,
        1,
        4
       ],
       "v4038": [
        2,
        16,
        1
       ],
       "v4039": [
        2,
        1,
        4
       ],
       "v4040": [
        2,
        16,
        1
       ],
       "v4041": [
        2,
        1,
        4
       ],
       "v4042": [
        2,
        16,
        1
       ],
       "v4043": [
        2,
        1,
        4
       ]
      },
      {
       "v115": [
        2,
        16,
        1
       ],
       "v116": [
        2,
        16,
        1
       ],
       "v118": [
        2,
        16,
        1
       ],
       "v119": [
        2,
        1,
        16
       ],
       "v120": [
        2,
        16,
        1
       ],
       "v121": [
        2,
        1,
        16
       ],
       "v4031": [
        2,
        16,
        1
       ],
       "v4032": [
        2,
        16,
        1
       ],
       "v4033": [
        2,
        16,
        1
       ],
       "v4034": [
        2,
        16,
        1
       ],
       "v4035": [
        2,
        16,
        16
       ],
       "v4036": [
        2,
        16,
        1
       ],
       "v4037": [
        2,
        1,
        16
       ],
       "v4038": [
        2,
        16,
        1
       ],
       "v4039": [
        2,
        1,
        16
       ],
       "v4040": [
        2,
        16,
        1
       ],
       "v4041": [
        2,
        1,
        16
       ],
       "v4042": [
        2,
        16,
        1
       ],
       "v4043": [
        2,
        1,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_4b31b4501a": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))), (mul 'v13', 'v14'))), (mul 'v15', 'v16'))), (mul 'v17', 'v18'))",
  "alpha_fold": "(synop2_35bbd03301 'v0', 'v1', 'v2', 'v3', (synop_bb8c54c201 'v4', (synop_bb8c54c201 'v5', 'v6', 'v7', 'v8'), 'v9', 'v10'), 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18')",
  "embeds": [
   "synop3_304457cd8d",
   "synop3_4cdc06b88f"
  ],
  "embeds_shipped": [
   "synop2_35bbd03301",
   "synop_bb8c54c201"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_4b31b4501a unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v548"
     },
     {
      "mvar": "v840"
     },
     {
      "mvar": "v841"
     },
     {
      "mvar": "v842"
     },
     {
      "mvar": "v843"
     },
     {
      "mvar": "v550"
     },
     {
      "mvar": "v551"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_4b31b4501a"
   },
   "missing_hooks": [],
   "name": "synop3_4b31b4501a.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v207"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v208"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v209"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v210"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v548"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v840"
                           },
                           {
                            "mvar": "v841"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v842"
                           },
                           {
                            "mvar": "v843"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v550"
                       },
                       {
                        "mvar": "v551"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v212"
                   },
                   {
                    "mvar": "v213"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v214"
               },
               {
                "mvar": "v215"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v216"
           },
           {
            "mvar": "v217"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v218"
       },
       {
        "mvar": "v219"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "args": [
       {
        "mvar": "v548"
       },
       {
        "args": [
         {
          "mvar": "v840"
         },
         {
          "mvar": "v841"
         },
         {
          "mvar": "v842"
         },
         {
          "mvar": "v843"
         }
        ],
        "attrs": {},
        "op": "synop_bb8c54c201"
       },
       {
        "mvar": "v550"
       },
       {
        "mvar": "v551"
       }
      ],
      "attrs": {},
      "op": "synop_bb8c54c201"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop2_35bbd03301"
   },
   "missing_hooks": [],
   "name": "prim:synop3_4b31b4501a",
   "rhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v548"
     },
     {
      "mvar": "v840"
     },
     {
      "mvar": "v841"
     },
     {
      "mvar": "v842"
     },
     {
      "mvar": "v843"
     },
     {
      "mvar": "v550"
     },
     {
      "mvar": "v551"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_4b31b4501a"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v207",
     "v208",
     "v209",
     "v210",
     "v548",
     "v840",
     "v841",
     "v842",
     "v843",
     "v550",
     "v551",
     "v212",
     "v213",
     "v214",
     "v215",
     "v216",
     "v217",
     "v218",
     "v219"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v207",
       [
        "add",
        [
         "mul",
         "v208",
         [
          "add",
          [
           "mul",
           "v209",
           [
            "add",
            [
             "mul",
             "v210",
             [
              "add",
              [
               "mul",
               "v548",
               [
                "add",
                [
                 "mul",
                 "v840",
                 "v841"
                ],
                [
                 "mul",
                 "v842",
                 "v843"
                ]
               ]
              ],
              [
               "mul",
               "v550",
               "v551"
              ]
             ]
            ],
            [
             "mul",
             "v212",
             "v213"
            ]
           ]
          ],
          [
           "mul",
           "v214",
           "v215"
          ]
         ]
        ],
        [
         "mul",
         "v216",
         "v217"
        ]
       ]
      ],
      [
       "mul",
       "v218",
       "v219"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v207",
      [
       "add",
       [
        "mul",
        "v208",
        [
         "add",
         [
          "mul",
          "v209",
          [
           "add",
           [
            "mul",
            "v210",
            [
             "add",
             [
              "mul",
              "v548",
              [
               "add",
               [
                "mul",
                "v840",
                "v841"
               ],
               [
                "mul",
                "v842",
                "v843"
               ]
              ]
             ],
             [
              "mul",
              "v550",
              "v551"
             ]
            ]
           ],
           [
            "mul",
            "v212",
            "v213"
           ]
          ]
         ],
         [
          "mul",
          "v214",
          "v215"
         ]
        ]
       ],
       [
        "mul",
        "v216",
        "v217"
       ]
      ]
     ],
     [
      "mul",
      "v218",
      "v219"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v548": [
        2,
        16,
        4
       ],
       "v550": [
        2,
        16,
        1
       ],
       "v551": [
        2,
        1,
        4
       ],
       "v840": [
        2,
        16,
        4
       ],
       "v841": [
        2,
        16,
        4
       ],
       "v842": [
        2,
        16,
        1
       ],
       "v843": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v548": [
        2,
        16,
        4
       ],
       "v550": [
        2,
        16,
        1
       ],
       "v551": [
        2,
        1,
        4
       ],
       "v840": [
        2,
        16,
        4
       ],
       "v841": [
        2,
        16,
        4
       ],
       "v842": [
        2,
        16,
        1
       ],
       "v843": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v548": [
        2,
        16,
        4
       ],
       "v550": [
        2,
        16,
        1
       ],
       "v551": [
        2,
        1,
        4
       ],
       "v840": [
        2,
        16,
        4
       ],
       "v841": [
        2,
        16,
        4
       ],
       "v842": [
        2,
        16,
        1
       ],
       "v843": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        1
       ],
       "v208": [
        2,
        16,
        1
       ],
       "v209": [
        2,
        16,
        1
       ],
       "v210": [
        2,
        16,
        1
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        16
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        16
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        16
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        16
       ],
       "v548": [
        2,
        16,
        1
       ],
       "v550": [
        2,
        16,
        1
       ],
       "v551": [
        2,
        1,
        16
       ],
       "v840": [
        2,
        16,
        1
       ],
       "v841": [
        2,
        16,
        16
       ],
       "v842": [
        2,
        16,
        1
       ],
       "v843": [
        2,
        1,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_4cdc06b88f": {
  "alpha": "(mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', 'v5'), (mul 'v6', 'v7'))), (mul 'v8', 'v9'))), (mul 'v10', 'v11'))), (mul 'v12', 'v13')))",
  "alpha_fold": "(mul 'v0', (synop2_d46808a970 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13'))",
  "embeds": [],
  "embeds_shipped": [
   "synop2_d46808a970"
  ],
  "equal_sites": 12,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_4cdc06b88f unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "mvar": "v336"
     },
     {
      "mvar": "v337"
     },
     {
      "mvar": "v338"
     },
     {
      "mvar": "v339"
     },
     {
      "mvar": "v340"
     },
     {
      "mvar": "v341"
     },
     {
      "mvar": "v342"
     },
     {
      "mvar": "v343"
     },
     {
      "mvar": "v344"
     },
     {
      "mvar": "v345"
     },
     {
      "mvar": "v346"
     },
     {
      "mvar": "v347"
     },
     {
      "mvar": "v348"
     }
    ],
    "attrs": {},
    "op": "synop3_4cdc06b88f"
   },
   "missing_hooks": [],
   "name": "synop3_4cdc06b88f.expand",
   "rhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v336"
         },
         {
          "args": [
           {
            "args": [
             {
              "mvar": "v337"
             },
             {
              "args": [
               {
                "args": [
                 {
                  "mvar": "v338"
                 },
                 {
                  "args": [
                   {
                    "args": [
                     {
                      "mvar": "v339"
                     },
                     {
                      "mvar": "v340"
                     }
                    ],
                    "attrs": {},
                    "op": "mul"
                   },
                   {
                    "args": [
                     {
                      "mvar": "v341"
                     },
                     {
                      "mvar": "v342"
                     }
                    ],
                    "attrs": {},
                    "op": "mul"
                   }
                  ],
                  "attrs": {},
                  "op": "add"
                 }
                ],
                "attrs": {},
                "op": "mul"
               },
               {
                "args": [
                 {
                  "mvar": "v343"
                 },
                 {
                  "mvar": "v344"
                 }
                ],
                "attrs": {},
                "op": "mul"
               }
              ],
              "attrs": {},
              "op": "add"
             }
            ],
            "attrs": {},
            "op": "mul"
           },
           {
            "args": [
             {
              "mvar": "v345"
             },
             {
              "mvar": "v346"
             }
            ],
            "attrs": {},
            "op": "mul"
           }
          ],
          "attrs": {},
          "op": "add"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "args": [
         {
          "mvar": "v347"
         },
         {
          "mvar": "v348"
         }
        ],
        "attrs": {},
        "op": "mul"
       }
      ],
      "attrs": {},
      "op": "add"
     }
    ],
    "attrs": {},
    "op": "mul"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "args": [
       {
        "mvar": "v336"
       },
       {
        "mvar": "v337"
       },
       {
        "mvar": "v338"
       },
       {
        "mvar": "v339"
       },
       {
        "mvar": "v340"
       },
       {
        "mvar": "v341"
       },
       {
        "mvar": "v342"
       },
       {
        "mvar": "v343"
       },
       {
        "mvar": "v344"
       },
       {
        "mvar": "v345"
       },
       {
        "mvar": "v346"
       },
       {
        "mvar": "v347"
       },
       {
        "mvar": "v348"
       }
      ],
      "attrs": {},
      "op": "synop2_d46808a970"
     }
    ],
    "attrs": {},
    "op": "mul"
   },
   "missing_hooks": [],
   "name": "prim:synop3_4cdc06b88f",
   "rhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "mvar": "v336"
     },
     {
      "mvar": "v337"
     },
     {
      "mvar": "v338"
     },
     {
      "mvar": "v339"
     },
     {
      "mvar": "v340"
     },
     {
      "mvar": "v341"
     },
     {
      "mvar": "v342"
     },
     {
      "mvar": "v343"
     },
     {
      "mvar": "v344"
     },
     {
      "mvar": "v345"
     },
     {
      "mvar": "v346"
     },
     {
      "mvar": "v347"
     },
     {
      "mvar": "v348"
     }
    ],
    "attrs": {},
    "op": "synop3_4cdc06b88f"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v10",
     "v336",
     "v337",
     "v338",
     "v339",
     "v340",
     "v341",
     "v342",
     "v343",
     "v344",
     "v345",
     "v346",
     "v347",
     "v348"
    ],
    "handler_body": [
     "ew",
     [
      "mul",
      "v10",
      [
       "add",
       [
        "mul",
        "v336",
        [
         "add",
         [
          "mul",
          "v337",
          [
           "add",
           [
            "mul",
            "v338",
            [
             "add",
             [
              "mul",
              "v339",
              "v340"
             ],
             [
              "mul",
              "v341",
              "v342"
             ]
            ]
           ],
           [
            "mul",
            "v343",
            "v344"
           ]
          ]
         ],
         [
          "mul",
          "v345",
          "v346"
         ]
        ]
       ],
       [
        "mul",
        "v347",
        "v348"
       ]
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "mul",
     "v10",
     [
      "add",
      [
       "mul",
       "v336",
       [
        "add",
        [
         "mul",
         "v337",
         [
          "add",
          [
           "mul",
           "v338",
           [
            "add",
            [
             "mul",
             "v339",
             "v340"
            ],
            [
             "mul",
             "v341",
             "v342"
            ]
           ]
          ],
          [
           "mul",
           "v343",
           "v344"
          ]
         ]
        ],
        [
         "mul",
         "v345",
         "v346"
        ]
       ]
      ],
      [
       "mul",
       "v347",
       "v348"
      ]
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v10": [
        2,
        16,
        4
       ],
       "v336": [
        2,
        16,
        4
       ],
       "v337": [
        2,
        16,
        4
       ],
       "v338": [
        2,
        16,
        4
       ],
       "v339": [
        2,
        16,
        4
       ],
       "v340": [
        2,
        16,
        4
       ],
       "v341": [
        2,
        16,
        1
       ],
       "v342": [
        2,
        1,
        4
       ],
       "v343": [
        2,
        16,
        1
       ],
       "v344": [
        2,
        1,
        4
       ],
       "v345": [
        2,
        16,
        1
       ],
       "v346": [
        2,
        1,
        4
       ],
       "v347": [
        2,
        16,
        1
       ],
       "v348": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v336": [
        2,
        16,
        4
       ],
       "v337": [
        2,
        16,
        4
       ],
       "v338": [
        2,
        16,
        4
       ],
       "v339": [
        2,
        16,
        4
       ],
       "v340": [
        2,
        16,
        4
       ],
       "v341": [
        2,
        16,
        1
       ],
       "v342": [
        2,
        1,
        4
       ],
       "v343": [
        2,
        16,
        1
       ],
       "v344": [
        2,
        1,
        4
       ],
       "v345": [
        2,
        16,
        1
       ],
       "v346": [
        2,
        1,
        4
       ],
       "v347": [
        2,
        16,
        1
       ],
       "v348": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v336": [
        2,
        16,
        4
       ],
       "v337": [
        2,
        16,
        4
       ],
       "v338": [
        2,
        16,
        4
       ],
       "v339": [
        2,
        16,
        4
       ],
       "v340": [
        2,
        16,
        4
       ],
       "v341": [
        2,
        16,
        1
       ],
       "v342": [
        2,
        1,
        4
       ],
       "v343": [
        2,
        16,
        1
       ],
       "v344": [
        2,
        1,
        4
       ],
       "v345": [
        2,
        16,
        1
       ],
       "v346": [
        2,
        1,
        4
       ],
       "v347": [
        2,
        16,
        1
       ],
       "v348": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v336": [
        2,
        16,
        4
       ],
       "v337": [
        2,
        16,
        4
       ],
       "v338": [
        2,
        16,
        4
       ],
       "v339": [
        2,
        16,
        4
       ],
       "v340": [
        2,
        16,
        4
       ],
       "v341": [
        2,
        16,
        1
       ],
       "v342": [
        2,
        1,
        4
       ],
       "v343": [
        2,
        16,
        1
       ],
       "v344": [
        2,
        1,
        4
       ],
       "v345": [
        2,
        16,
        1
       ],
       "v346": [
        2,
        1,
        4
       ],
       "v347": [
        2,
        16,
        1
       ],
       "v348": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 12
 },
 "synop3_4cf8732640": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))), (mul 'v13', 'v14'))), (mul 'v15', 'v16'))), (mul 'v17', 'v18'))",
  "alpha_fold": "(synop2_35bbd03301 'v0', 'v1', 'v2', 'v3', (synop_6c709f774a 'v4', 'v5', 'v6', 'v7', 'v8', (mul 'v9', 'v10')), 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18')",
  "embeds": [
   "synop3_304457cd8d",
   "synop3_4cdc06b88f"
  ],
  "embeds_shipped": [
   "synop2_35bbd03301",
   "synop_6c709f774a"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_4cf8732640 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v530"
     },
     {
      "mvar": "v531"
     },
     {
      "mvar": "v532"
     },
     {
      "mvar": "v533"
     },
     {
      "mvar": "v534"
     },
     {
      "mvar": "v2937"
     },
     {
      "mvar": "v2938"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_4cf8732640"
   },
   "missing_hooks": [],
   "name": "synop3_4cf8732640.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v207"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v208"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v209"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v210"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v530"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v531"
                           },
                           {
                            "mvar": "v532"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v533"
                           },
                           {
                            "mvar": "v534"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v2937"
                       },
                       {
                        "mvar": "v2938"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v212"
                   },
                   {
                    "mvar": "v213"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v214"
               },
               {
                "mvar": "v215"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v216"
           },
           {
            "mvar": "v217"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v218"
       },
       {
        "mvar": "v219"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "args": [
       {
        "mvar": "v530"
       },
       {
        "mvar": "v531"
       },
       {
        "mvar": "v532"
       },
       {
        "mvar": "v533"
       },
       {
        "mvar": "v534"
       },
       {
        "args": [
         {
          "mvar": "v2937"
         },
         {
          "mvar": "v2938"
         }
        ],
        "attrs": {},
        "op": "mul"
       }
      ],
      "attrs": {},
      "op": "synop_6c709f774a"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop2_35bbd03301"
   },
   "missing_hooks": [],
   "name": "prim:synop3_4cf8732640",
   "rhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v530"
     },
     {
      "mvar": "v531"
     },
     {
      "mvar": "v532"
     },
     {
      "mvar": "v533"
     },
     {
      "mvar": "v534"
     },
     {
      "mvar": "v2937"
     },
     {
      "mvar": "v2938"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_4cf8732640"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v207",
     "v208",
     "v209",
     "v210",
     "v530",
     "v531",
     "v532",
     "v533",
     "v534",
     "v2937",
     "v2938",
     "v212",
     "v213",
     "v214",
     "v215",
     "v216",
     "v217",
     "v218",
     "v219"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v207",
       [
        "add",
        [
         "mul",
         "v208",
         [
          "add",
          [
           "mul",
           "v209",
           [
            "add",
            [
             "mul",
             "v210",
             [
              "add",
              [
               "mul",
               "v530",
               [
                "add",
                [
                 "mul",
                 "v531",
                 "v532"
                ],
                [
                 "mul",
                 "v533",
                 "v534"
                ]
               ]
              ],
              [
               "mul",
               "v2937",
               "v2938"
              ]
             ]
            ],
            [
             "mul",
             "v212",
             "v213"
            ]
           ]
          ],
          [
           "mul",
           "v214",
           "v215"
          ]
         ]
        ],
        [
         "mul",
         "v216",
         "v217"
        ]
       ]
      ],
      [
       "mul",
       "v218",
       "v219"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v207",
      [
       "add",
       [
        "mul",
        "v208",
        [
         "add",
         [
          "mul",
          "v209",
          [
           "add",
           [
            "mul",
            "v210",
            [
             "add",
             [
              "mul",
              "v530",
              [
               "add",
               [
                "mul",
                "v531",
                "v532"
               ],
               [
                "mul",
                "v533",
                "v534"
               ]
              ]
             ],
             [
              "mul",
              "v2937",
              "v2938"
             ]
            ]
           ],
           [
            "mul",
            "v212",
            "v213"
           ]
          ]
         ],
         [
          "mul",
          "v214",
          "v215"
         ]
        ]
       ],
       [
        "mul",
        "v216",
        "v217"
       ]
      ]
     ],
     [
      "mul",
      "v218",
      "v219"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v2937": [
        2,
        16,
        1
       ],
       "v2938": [
        2,
        1,
        4
       ],
       "v530": [
        2,
        16,
        4
       ],
       "v531": [
        2,
        16,
        4
       ],
       "v532": [
        2,
        16,
        4
       ],
       "v533": [
        2,
        16,
        1
       ],
       "v534": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v2937": [
        2,
        16,
        1
       ],
       "v2938": [
        2,
        1,
        4
       ],
       "v530": [
        2,
        16,
        4
       ],
       "v531": [
        2,
        16,
        4
       ],
       "v532": [
        2,
        16,
        4
       ],
       "v533": [
        2,
        16,
        1
       ],
       "v534": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v2937": [
        2,
        16,
        1
       ],
       "v2938": [
        2,
        1,
        4
       ],
       "v530": [
        2,
        16,
        4
       ],
       "v531": [
        2,
        16,
        4
       ],
       "v532": [
        2,
        16,
        4
       ],
       "v533": [
        2,
        16,
        1
       ],
       "v534": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        1
       ],
       "v208": [
        2,
        16,
        1
       ],
       "v209": [
        2,
        16,
        1
       ],
       "v210": [
        2,
        16,
        1
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        16
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        16
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        16
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        16
       ],
       "v2937": [
        2,
        16,
        1
       ],
       "v2938": [
        2,
        1,
        16
       ],
       "v530": [
        2,
        16,
        1
       ],
       "v531": [
        2,
        16,
        1
       ],
       "v532": [
        2,
        16,
        16
       ],
       "v533": [
        2,
        16,
        1
       ],
       "v534": [
        2,
        1,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_50a12a0158": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))), (mul 'v13', 'v14'))), (mul 'v15', 'v16'))), (mul 'v17', 'v18'))",
  "alpha_fold": "(synop2_35bbd03301 'v0', 'v1', 'v2', 'v3', (synop_f3290a1163 'v4', 'v5', 'v6', (mul 'v7', 'v8'), 'v9', 'v10'), 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18')",
  "embeds": [
   "synop3_304457cd8d",
   "synop3_4cdc06b88f"
  ],
  "embeds_shipped": [
   "synop2_35bbd03301",
   "synop_f3290a1163"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_50a12a0158 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v565"
     },
     {
      "mvar": "v566"
     },
     {
      "mvar": "v567"
     },
     {
      "mvar": "v3133"
     },
     {
      "mvar": "v3134"
     },
     {
      "mvar": "v569"
     },
     {
      "mvar": "v570"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_50a12a0158"
   },
   "missing_hooks": [],
   "name": "synop3_50a12a0158.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v207"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v208"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v209"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v210"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v565"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v566"
                           },
                           {
                            "mvar": "v567"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v3133"
                           },
                           {
                            "mvar": "v3134"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v569"
                       },
                       {
                        "mvar": "v570"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v212"
                   },
                   {
                    "mvar": "v213"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v214"
               },
               {
                "mvar": "v215"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v216"
           },
           {
            "mvar": "v217"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v218"
       },
       {
        "mvar": "v219"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "args": [
       {
        "mvar": "v565"
       },
       {
        "mvar": "v566"
       },
       {
        "mvar": "v567"
       },
       {
        "args": [
         {
          "mvar": "v3133"
         },
         {
          "mvar": "v3134"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "mvar": "v569"
       },
       {
        "mvar": "v570"
       }
      ],
      "attrs": {},
      "op": "synop_f3290a1163"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop2_35bbd03301"
   },
   "missing_hooks": [],
   "name": "prim:synop3_50a12a0158",
   "rhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v565"
     },
     {
      "mvar": "v566"
     },
     {
      "mvar": "v567"
     },
     {
      "mvar": "v3133"
     },
     {
      "mvar": "v3134"
     },
     {
      "mvar": "v569"
     },
     {
      "mvar": "v570"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_50a12a0158"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v207",
     "v208",
     "v209",
     "v210",
     "v565",
     "v566",
     "v567",
     "v3133",
     "v3134",
     "v569",
     "v570",
     "v212",
     "v213",
     "v214",
     "v215",
     "v216",
     "v217",
     "v218",
     "v219"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v207",
       [
        "add",
        [
         "mul",
         "v208",
         [
          "add",
          [
           "mul",
           "v209",
           [
            "add",
            [
             "mul",
             "v210",
             [
              "add",
              [
               "mul",
               "v565",
               [
                "add",
                [
                 "mul",
                 "v566",
                 "v567"
                ],
                [
                 "mul",
                 "v3133",
                 "v3134"
                ]
               ]
              ],
              [
               "mul",
               "v569",
               "v570"
              ]
             ]
            ],
            [
             "mul",
             "v212",
             "v213"
            ]
           ]
          ],
          [
           "mul",
           "v214",
           "v215"
          ]
         ]
        ],
        [
         "mul",
         "v216",
         "v217"
        ]
       ]
      ],
      [
       "mul",
       "v218",
       "v219"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v207",
      [
       "add",
       [
        "mul",
        "v208",
        [
         "add",
         [
          "mul",
          "v209",
          [
           "add",
           [
            "mul",
            "v210",
            [
             "add",
             [
              "mul",
              "v565",
              [
               "add",
               [
                "mul",
                "v566",
                "v567"
               ],
               [
                "mul",
                "v3133",
                "v3134"
               ]
              ]
             ],
             [
              "mul",
              "v569",
              "v570"
             ]
            ]
           ],
           [
            "mul",
            "v212",
            "v213"
           ]
          ]
         ],
         [
          "mul",
          "v214",
          "v215"
         ]
        ]
       ],
       [
        "mul",
        "v216",
        "v217"
       ]
      ]
     ],
     [
      "mul",
      "v218",
      "v219"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v3133": [
        2,
        16,
        1
       ],
       "v3134": [
        2,
        1,
        4
       ],
       "v565": [
        2,
        16,
        4
       ],
       "v566": [
        2,
        16,
        4
       ],
       "v567": [
        2,
        16,
        4
       ],
       "v569": [
        2,
        16,
        1
       ],
       "v570": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v3133": [
        2,
        16,
        1
       ],
       "v3134": [
        2,
        1,
        4
       ],
       "v565": [
        2,
        16,
        4
       ],
       "v566": [
        2,
        16,
        4
       ],
       "v567": [
        2,
        16,
        4
       ],
       "v569": [
        2,
        16,
        1
       ],
       "v570": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v3133": [
        2,
        16,
        1
       ],
       "v3134": [
        2,
        1,
        4
       ],
       "v565": [
        2,
        16,
        4
       ],
       "v566": [
        2,
        16,
        4
       ],
       "v567": [
        2,
        16,
        4
       ],
       "v569": [
        2,
        16,
        1
       ],
       "v570": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        1
       ],
       "v208": [
        2,
        16,
        1
       ],
       "v209": [
        2,
        16,
        1
       ],
       "v210": [
        2,
        16,
        1
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        16
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        16
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        16
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        16
       ],
       "v3133": [
        2,
        16,
        1
       ],
       "v3134": [
        2,
        1,
        16
       ],
       "v565": [
        2,
        16,
        1
       ],
       "v566": [
        2,
        16,
        1
       ],
       "v567": [
        2,
        16,
        16
       ],
       "v569": [
        2,
        16,
        1
       ],
       "v570": [
        2,
        1,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_531bb1872b": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))), (mul 'v13', 'v14'))), (mul 'v15', 'v16'))), (mul 'v17', 'v18'))",
  "alpha_fold": "(synop2_35bbd03301 'v0', 'v1', 'v2', 'v3', (add (mul 'v4', (synop_bb8c54c201 'v5', 'v6', 'v7', 'v8')), (mul 'v9', 'v10')), 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18')",
  "embeds": [
   "synop3_304457cd8d",
   "synop3_4cdc06b88f"
  ],
  "embeds_shipped": [
   "synop2_35bbd03301",
   "synop_bb8c54c201"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_531bb1872b unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v1681"
     },
     {
      "mvar": "v1951"
     },
     {
      "mvar": "v1952"
     },
     {
      "mvar": "v1953"
     },
     {
      "mvar": "v1954"
     },
     {
      "mvar": "v1752"
     },
     {
      "mvar": "v1753"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_531bb1872b"
   },
   "missing_hooks": [],
   "name": "synop3_531bb1872b.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v207"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v208"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v209"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v210"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v1681"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v1951"
                           },
                           {
                            "mvar": "v1952"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v1953"
                           },
                           {
                            "mvar": "v1954"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v1752"
                       },
                       {
                        "mvar": "v1753"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v212"
                   },
                   {
                    "mvar": "v213"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v214"
               },
               {
                "mvar": "v215"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v216"
           },
           {
            "mvar": "v217"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v218"
       },
       {
        "mvar": "v219"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v1681"
         },
         {
          "args": [
           {
            "mvar": "v1951"
           },
           {
            "mvar": "v1952"
           },
           {
            "mvar": "v1953"
           },
           {
            "mvar": "v1954"
           }
          ],
          "attrs": {},
          "op": "synop_bb8c54c201"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "args": [
         {
          "mvar": "v1752"
         },
         {
          "mvar": "v1753"
         }
        ],
        "attrs": {},
        "op": "mul"
       }
      ],
      "attrs": {},
      "op": "add"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop2_35bbd03301"
   },
   "missing_hooks": [],
   "name": "prim:synop3_531bb1872b",
   "rhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v1681"
     },
     {
      "mvar": "v1951"
     },
     {
      "mvar": "v1952"
     },
     {
      "mvar": "v1953"
     },
     {
      "mvar": "v1954"
     },
     {
      "mvar": "v1752"
     },
     {
      "mvar": "v1753"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_531bb1872b"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v207",
     "v208",
     "v209",
     "v210",
     "v1681",
     "v1951",
     "v1952",
     "v1953",
     "v1954",
     "v1752",
     "v1753",
     "v212",
     "v213",
     "v214",
     "v215",
     "v216",
     "v217",
     "v218",
     "v219"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v207",
       [
        "add",
        [
         "mul",
         "v208",
         [
          "add",
          [
           "mul",
           "v209",
           [
            "add",
            [
             "mul",
             "v210",
             [
              "add",
              [
               "mul",
               "v1681",
               [
                "add",
                [
                 "mul",
                 "v1951",
                 "v1952"
                ],
                [
                 "mul",
                 "v1953",
                 "v1954"
                ]
               ]
              ],
              [
               "mul",
               "v1752",
               "v1753"
              ]
             ]
            ],
            [
             "mul",
             "v212",
             "v213"
            ]
           ]
          ],
          [
           "mul",
           "v214",
           "v215"
          ]
         ]
        ],
        [
         "mul",
         "v216",
         "v217"
        ]
       ]
      ],
      [
       "mul",
       "v218",
       "v219"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v207",
      [
       "add",
       [
        "mul",
        "v208",
        [
         "add",
         [
          "mul",
          "v209",
          [
           "add",
           [
            "mul",
            "v210",
            [
             "add",
             [
              "mul",
              "v1681",
              [
               "add",
               [
                "mul",
                "v1951",
                "v1952"
               ],
               [
                "mul",
                "v1953",
                "v1954"
               ]
              ]
             ],
             [
              "mul",
              "v1752",
              "v1753"
             ]
            ]
           ],
           [
            "mul",
            "v212",
            "v213"
           ]
          ]
         ],
         [
          "mul",
          "v214",
          "v215"
         ]
        ]
       ],
       [
        "mul",
        "v216",
        "v217"
       ]
      ]
     ],
     [
      "mul",
      "v218",
      "v219"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v1681": [
        2,
        16,
        4
       ],
       "v1752": [
        2,
        16,
        1
       ],
       "v1753": [
        2,
        1,
        4
       ],
       "v1951": [
        2,
        16,
        4
       ],
       "v1952": [
        2,
        16,
        4
       ],
       "v1953": [
        2,
        16,
        1
       ],
       "v1954": [
        2,
        1,
        4
       ],
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ]
      },
      {
       "v1681": [
        2,
        16,
        4
       ],
       "v1752": [
        2,
        16,
        1
       ],
       "v1753": [
        2,
        1,
        4
       ],
       "v1951": [
        2,
        16,
        4
       ],
       "v1952": [
        2,
        16,
        4
       ],
       "v1953": [
        2,
        16,
        1
       ],
       "v1954": [
        2,
        1,
        4
       ],
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ]
      },
      {
       "v1681": [
        2,
        16,
        4
       ],
       "v1752": [
        2,
        16,
        1
       ],
       "v1753": [
        2,
        1,
        4
       ],
       "v1951": [
        2,
        16,
        4
       ],
       "v1952": [
        2,
        16,
        4
       ],
       "v1953": [
        2,
        16,
        1
       ],
       "v1954": [
        2,
        1,
        4
       ],
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ]
      },
      {
       "v1681": [
        2,
        16,
        1
       ],
       "v1752": [
        2,
        16,
        1
       ],
       "v1753": [
        2,
        1,
        16
       ],
       "v1951": [
        2,
        16,
        1
       ],
       "v1952": [
        2,
        16,
        16
       ],
       "v1953": [
        2,
        16,
        1
       ],
       "v1954": [
        2,
        1,
        16
       ],
       "v207": [
        2,
        16,
        1
       ],
       "v208": [
        2,
        16,
        1
       ],
       "v209": [
        2,
        16,
        1
       ],
       "v210": [
        2,
        16,
        1
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        16
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        16
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        16
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_543ad6d861": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), (mul 'v7', 'v8'))), 'v9')), (mul 'v10', 'v11'))), (mul 'v12', 'v13'))), (mul 'v14', 'v15'))), (mul 'v16', 'v17'))",
  "alpha_fold": "(synop2_35bbd03301 'v0', 'v1', 'v2', 'v3', (add (mul 'v4', (synop_bb8c54c201 'v5', 'v6', 'v7', 'v8')), 'v9'), 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17')",
  "embeds": [],
  "embeds_shipped": [
   "synop2_35bbd03301",
   "synop_bb8c54c201"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_543ad6d861 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v1681"
     },
     {
      "mvar": "v1827"
     },
     {
      "mvar": "v1828"
     },
     {
      "mvar": "v1829"
     },
     {
      "mvar": "v1830"
     },
     {
      "mvar": "v452"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_543ad6d861"
   },
   "missing_hooks": [],
   "name": "synop3_543ad6d861.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v207"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v208"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v209"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v210"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v1681"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v1827"
                           },
                           {
                            "mvar": "v1828"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v1829"
                           },
                           {
                            "mvar": "v1830"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "mvar": "v452"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v212"
                   },
                   {
                    "mvar": "v213"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v214"
               },
               {
                "mvar": "v215"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v216"
           },
           {
            "mvar": "v217"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v218"
       },
       {
        "mvar": "v219"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v1681"
         },
         {
          "args": [
           {
            "mvar": "v1827"
           },
           {
            "mvar": "v1828"
           },
           {
            "mvar": "v1829"
           },
           {
            "mvar": "v1830"
           }
          ],
          "attrs": {},
          "op": "synop_bb8c54c201"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "mvar": "v452"
       }
      ],
      "attrs": {},
      "op": "add"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop2_35bbd03301"
   },
   "missing_hooks": [],
   "name": "prim:synop3_543ad6d861",
   "rhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v1681"
     },
     {
      "mvar": "v1827"
     },
     {
      "mvar": "v1828"
     },
     {
      "mvar": "v1829"
     },
     {
      "mvar": "v1830"
     },
     {
      "mvar": "v452"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_543ad6d861"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v207",
     "v208",
     "v209",
     "v210",
     "v1681",
     "v1827",
     "v1828",
     "v1829",
     "v1830",
     "v452",
     "v212",
     "v213",
     "v214",
     "v215",
     "v216",
     "v217",
     "v218",
     "v219"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v207",
       [
        "add",
        [
         "mul",
         "v208",
         [
          "add",
          [
           "mul",
           "v209",
           [
            "add",
            [
             "mul",
             "v210",
             [
              "add",
              [
               "mul",
               "v1681",
               [
                "add",
                [
                 "mul",
                 "v1827",
                 "v1828"
                ],
                [
                 "mul",
                 "v1829",
                 "v1830"
                ]
               ]
              ],
              "v452"
             ]
            ],
            [
             "mul",
             "v212",
             "v213"
            ]
           ]
          ],
          [
           "mul",
           "v214",
           "v215"
          ]
         ]
        ],
        [
         "mul",
         "v216",
         "v217"
        ]
       ]
      ],
      [
       "mul",
       "v218",
       "v219"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v207",
      [
       "add",
       [
        "mul",
        "v208",
        [
         "add",
         [
          "mul",
          "v209",
          [
           "add",
           [
            "mul",
            "v210",
            [
             "add",
             [
              "mul",
              "v1681",
              [
               "add",
               [
                "mul",
                "v1827",
                "v1828"
               ],
               [
                "mul",
                "v1829",
                "v1830"
               ]
              ]
             ],
             "v452"
            ]
           ],
           [
            "mul",
            "v212",
            "v213"
           ]
          ]
         ],
         [
          "mul",
          "v214",
          "v215"
         ]
        ]
       ],
       [
        "mul",
        "v216",
        "v217"
       ]
      ]
     ],
     [
      "mul",
      "v218",
      "v219"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v1681": [
        2,
        16,
        4
       ],
       "v1827": [
        2,
        16,
        4
       ],
       "v1828": [
        2,
        16,
        4
       ],
       "v1829": [
        2,
        16,
        1
       ],
       "v1830": [
        2,
        1,
        4
       ],
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v452": [
        2,
        16,
        4
       ]
      },
      {
       "v1681": [
        2,
        16,
        4
       ],
       "v1827": [
        2,
        16,
        4
       ],
       "v1828": [
        2,
        16,
        4
       ],
       "v1829": [
        2,
        16,
        1
       ],
       "v1830": [
        2,
        1,
        4
       ],
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v452": [
        2,
        16,
        4
       ]
      },
      {
       "v1681": [
        2,
        16,
        4
       ],
       "v1827": [
        2,
        16,
        4
       ],
       "v1828": [
        2,
        16,
        4
       ],
       "v1829": [
        2,
        16,
        1
       ],
       "v1830": [
        2,
        1,
        4
       ],
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v452": [
        2,
        16,
        4
       ]
      },
      {
       "v1681": [
        2,
        16,
        1
       ],
       "v1827": [
        2,
        16,
        1
       ],
       "v1828": [
        2,
        16,
        16
       ],
       "v1829": [
        2,
        16,
        1
       ],
       "v1830": [
        2,
        1,
        16
       ],
       "v207": [
        2,
        16,
        1
       ],
       "v208": [
        2,
        16,
        1
       ],
       "v209": [
        2,
        16,
        1
       ],
       "v210": [
        2,
        16,
        1
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        16
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        16
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        16
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        16
       ],
       "v452": [
        2,
        16,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_970223f613": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), (mul 'v7', 'v8'))), 'v9')), (mul 'v10', 'v11'))), (mul 'v12', 'v13'))), (mul 'v14', 'v15'))), (mul 'v16', 'v17'))",
  "alpha_fold": "(synop2_35bbd03301 'v0', 'v1', 'v2', 'v3', (synop_6c709f774a 'v4', 'v5', 'v6', 'v7', 'v8', 'v9'), 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17')",
  "embeds": [],
  "embeds_shipped": [
   "synop2_35bbd03301",
   "synop_6c709f774a"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_970223f613 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v530"
     },
     {
      "mvar": "v531"
     },
     {
      "mvar": "v532"
     },
     {
      "mvar": "v533"
     },
     {
      "mvar": "v534"
     },
     {
      "mvar": "v535"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_970223f613"
   },
   "missing_hooks": [],
   "name": "synop3_970223f613.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v207"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v208"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v209"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v210"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v530"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v531"
                           },
                           {
                            "mvar": "v532"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v533"
                           },
                           {
                            "mvar": "v534"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "mvar": "v535"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v212"
                   },
                   {
                    "mvar": "v213"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v214"
               },
               {
                "mvar": "v215"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v216"
           },
           {
            "mvar": "v217"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v218"
       },
       {
        "mvar": "v219"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "args": [
       {
        "mvar": "v530"
       },
       {
        "mvar": "v531"
       },
       {
        "mvar": "v532"
       },
       {
        "mvar": "v533"
       },
       {
        "mvar": "v534"
       },
       {
        "mvar": "v535"
       }
      ],
      "attrs": {},
      "op": "synop_6c709f774a"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop2_35bbd03301"
   },
   "missing_hooks": [],
   "name": "prim:synop3_970223f613",
   "rhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v530"
     },
     {
      "mvar": "v531"
     },
     {
      "mvar": "v532"
     },
     {
      "mvar": "v533"
     },
     {
      "mvar": "v534"
     },
     {
      "mvar": "v535"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_970223f613"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v207",
     "v208",
     "v209",
     "v210",
     "v530",
     "v531",
     "v532",
     "v533",
     "v534",
     "v535",
     "v212",
     "v213",
     "v214",
     "v215",
     "v216",
     "v217",
     "v218",
     "v219"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v207",
       [
        "add",
        [
         "mul",
         "v208",
         [
          "add",
          [
           "mul",
           "v209",
           [
            "add",
            [
             "mul",
             "v210",
             [
              "add",
              [
               "mul",
               "v530",
               [
                "add",
                [
                 "mul",
                 "v531",
                 "v532"
                ],
                [
                 "mul",
                 "v533",
                 "v534"
                ]
               ]
              ],
              "v535"
             ]
            ],
            [
             "mul",
             "v212",
             "v213"
            ]
           ]
          ],
          [
           "mul",
           "v214",
           "v215"
          ]
         ]
        ],
        [
         "mul",
         "v216",
         "v217"
        ]
       ]
      ],
      [
       "mul",
       "v218",
       "v219"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v207",
      [
       "add",
       [
        "mul",
        "v208",
        [
         "add",
         [
          "mul",
          "v209",
          [
           "add",
           [
            "mul",
            "v210",
            [
             "add",
             [
              "mul",
              "v530",
              [
               "add",
               [
                "mul",
                "v531",
                "v532"
               ],
               [
                "mul",
                "v533",
                "v534"
               ]
              ]
             ],
             "v535"
            ]
           ],
           [
            "mul",
            "v212",
            "v213"
           ]
          ]
         ],
         [
          "mul",
          "v214",
          "v215"
         ]
        ]
       ],
       [
        "mul",
        "v216",
        "v217"
       ]
      ]
     ],
     [
      "mul",
      "v218",
      "v219"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v530": [
        2,
        16,
        4
       ],
       "v531": [
        2,
        16,
        4
       ],
       "v532": [
        2,
        16,
        4
       ],
       "v533": [
        2,
        16,
        1
       ],
       "v534": [
        2,
        1,
        4
       ],
       "v535": [
        2,
        16,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v530": [
        2,
        16,
        4
       ],
       "v531": [
        2,
        16,
        4
       ],
       "v532": [
        2,
        16,
        4
       ],
       "v533": [
        2,
        16,
        1
       ],
       "v534": [
        2,
        1,
        4
       ],
       "v535": [
        2,
        16,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v530": [
        2,
        16,
        4
       ],
       "v531": [
        2,
        16,
        4
       ],
       "v532": [
        2,
        16,
        4
       ],
       "v533": [
        2,
        16,
        1
       ],
       "v534": [
        2,
        1,
        4
       ],
       "v535": [
        2,
        16,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        1
       ],
       "v208": [
        2,
        16,
        1
       ],
       "v209": [
        2,
        16,
        1
       ],
       "v210": [
        2,
        16,
        1
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        16
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        16
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        16
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        16
       ],
       "v530": [
        2,
        16,
        1
       ],
       "v531": [
        2,
        16,
        1
       ],
       "v532": [
        2,
        16,
        16
       ],
       "v533": [
        2,
        16,
        1
       ],
       "v534": [
        2,
        1,
        16
       ],
       "v535": [
        2,
        16,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_9b8631df70": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', 'v5'), (mul 'v6', 'v7'))), (mul 'v8', 'v9'))), (mul 'v10', 'v11'))), (mul 'v12', 'v13'))), (mul 'v14', 'v15'))",
  "alpha_fold": "(synop2_d46808a970 'v0', 'v1', 'v2', 'v3', (add (mul 'v4', 'v5'), (mul 'v6', 'v7')), 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15')",
  "embeds": [
   "synop3_4cdc06b88f"
  ],
  "embeds_shipped": [
   "synop2_d46808a970"
  ],
  "equal_sites": 12,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_9b8631df70 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v2183"
     },
     {
      "mvar": "v2184"
     },
     {
      "mvar": "v2254"
     },
     {
      "mvar": "v2255"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_9b8631df70"
   },
   "missing_hooks": [],
   "name": "synop3_9b8631df70.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v220"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v221"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v222"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v223"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v2183"
                       },
                       {
                        "mvar": "v2184"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v2254"
                       },
                       {
                        "mvar": "v2255"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v225"
                   },
                   {
                    "mvar": "v226"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v227"
               },
               {
                "mvar": "v228"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v229"
           },
           {
            "mvar": "v230"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v231"
       },
       {
        "mvar": "v232"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v2183"
         },
         {
          "mvar": "v2184"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "args": [
         {
          "mvar": "v2254"
         },
         {
          "mvar": "v2255"
         }
        ],
        "attrs": {},
        "op": "mul"
       }
      ],
      "attrs": {},
      "op": "add"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop2_d46808a970"
   },
   "missing_hooks": [],
   "name": "prim:synop3_9b8631df70",
   "rhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v2183"
     },
     {
      "mvar": "v2184"
     },
     {
      "mvar": "v2254"
     },
     {
      "mvar": "v2255"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_9b8631df70"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v220",
     "v221",
     "v222",
     "v223",
     "v2183",
     "v2184",
     "v2254",
     "v2255",
     "v225",
     "v226",
     "v227",
     "v228",
     "v229",
     "v230",
     "v231",
     "v232"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v220",
       [
        "add",
        [
         "mul",
         "v221",
         [
          "add",
          [
           "mul",
           "v222",
           [
            "add",
            [
             "mul",
             "v223",
             [
              "add",
              [
               "mul",
               "v2183",
               "v2184"
              ],
              [
               "mul",
               "v2254",
               "v2255"
              ]
             ]
            ],
            [
             "mul",
             "v225",
             "v226"
            ]
           ]
          ],
          [
           "mul",
           "v227",
           "v228"
          ]
         ]
        ],
        [
         "mul",
         "v229",
         "v230"
        ]
       ]
      ],
      [
       "mul",
       "v231",
       "v232"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v220",
      [
       "add",
       [
        "mul",
        "v221",
        [
         "add",
         [
          "mul",
          "v222",
          [
           "add",
           [
            "mul",
            "v223",
            [
             "add",
             [
              "mul",
              "v2183",
              "v2184"
             ],
             [
              "mul",
              "v2254",
              "v2255"
             ]
            ]
           ],
           [
            "mul",
            "v225",
            "v226"
           ]
          ]
         ],
         [
          "mul",
          "v227",
          "v228"
         ]
        ]
       ],
       [
        "mul",
        "v229",
        "v230"
       ]
      ]
     ],
     [
      "mul",
      "v231",
      "v232"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v2183": [
        2,
        16,
        4
       ],
       "v2184": [
        2,
        16,
        4
       ],
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v2254": [
        2,
        16,
        1
       ],
       "v2255": [
        2,
        1,
        4
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ]
      },
      {
       "v2183": [
        2,
        16,
        4
       ],
       "v2184": [
        2,
        16,
        4
       ],
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v2254": [
        2,
        16,
        1
       ],
       "v2255": [
        2,
        1,
        4
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ]
      },
      {
       "v2183": [
        2,
        16,
        4
       ],
       "v2184": [
        2,
        16,
        4
       ],
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v2254": [
        2,
        16,
        1
       ],
       "v2255": [
        2,
        1,
        4
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ]
      },
      {
       "v2183": [
        2,
        16,
        4
       ],
       "v2184": [
        2,
        16,
        4
       ],
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v2254": [
        2,
        16,
        1
       ],
       "v2255": [
        2,
        1,
        4
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 12
 },
 "synop3_acc3c38681": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))), (mul 'v13', 'v14'))), (mul 'v15', 'v16'))), (mul 'v17', 'v18'))",
  "alpha_fold": "(synop2_35bbd03301 'v0', 'v1', 'v2', 'v3', (synop_2b8cfd1c90 'v4', (mul 'v5', 'v6'), 'v7', 'v8', 'v9', 'v10'), 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18')",
  "embeds": [
   "synop3_304457cd8d",
   "synop3_4cdc06b88f"
  ],
  "embeds_shipped": [
   "synop2_35bbd03301",
   "synop_2b8cfd1c90"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_acc3c38681 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v485"
     },
     {
      "mvar": "v2687"
     },
     {
      "mvar": "v2688"
     },
     {
      "mvar": "v487"
     },
     {
      "mvar": "v488"
     },
     {
      "mvar": "v489"
     },
     {
      "mvar": "v490"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_acc3c38681"
   },
   "missing_hooks": [],
   "name": "synop3_acc3c38681.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v207"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v208"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v209"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v210"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v485"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v2687"
                           },
                           {
                            "mvar": "v2688"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v487"
                           },
                           {
                            "mvar": "v488"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v489"
                       },
                       {
                        "mvar": "v490"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v212"
                   },
                   {
                    "mvar": "v213"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v214"
               },
               {
                "mvar": "v215"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v216"
           },
           {
            "mvar": "v217"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v218"
       },
       {
        "mvar": "v219"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "args": [
       {
        "mvar": "v485"
       },
       {
        "args": [
         {
          "mvar": "v2687"
         },
         {
          "mvar": "v2688"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "mvar": "v487"
       },
       {
        "mvar": "v488"
       },
       {
        "mvar": "v489"
       },
       {
        "mvar": "v490"
       }
      ],
      "attrs": {},
      "op": "synop_2b8cfd1c90"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop2_35bbd03301"
   },
   "missing_hooks": [],
   "name": "prim:synop3_acc3c38681",
   "rhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v485"
     },
     {
      "mvar": "v2687"
     },
     {
      "mvar": "v2688"
     },
     {
      "mvar": "v487"
     },
     {
      "mvar": "v488"
     },
     {
      "mvar": "v489"
     },
     {
      "mvar": "v490"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_acc3c38681"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v207",
     "v208",
     "v209",
     "v210",
     "v485",
     "v2687",
     "v2688",
     "v487",
     "v488",
     "v489",
     "v490",
     "v212",
     "v213",
     "v214",
     "v215",
     "v216",
     "v217",
     "v218",
     "v219"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v207",
       [
        "add",
        [
         "mul",
         "v208",
         [
          "add",
          [
           "mul",
           "v209",
           [
            "add",
            [
             "mul",
             "v210",
             [
              "add",
              [
               "mul",
               "v485",
               [
                "add",
                [
                 "mul",
                 "v2687",
                 "v2688"
                ],
                [
                 "mul",
                 "v487",
                 "v488"
                ]
               ]
              ],
              [
               "mul",
               "v489",
               "v490"
              ]
             ]
            ],
            [
             "mul",
             "v212",
             "v213"
            ]
           ]
          ],
          [
           "mul",
           "v214",
           "v215"
          ]
         ]
        ],
        [
         "mul",
         "v216",
         "v217"
        ]
       ]
      ],
      [
       "mul",
       "v218",
       "v219"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v207",
      [
       "add",
       [
        "mul",
        "v208",
        [
         "add",
         [
          "mul",
          "v209",
          [
           "add",
           [
            "mul",
            "v210",
            [
             "add",
             [
              "mul",
              "v485",
              [
               "add",
               [
                "mul",
                "v2687",
                "v2688"
               ],
               [
                "mul",
                "v487",
                "v488"
               ]
              ]
             ],
             [
              "mul",
              "v489",
              "v490"
             ]
            ]
           ],
           [
            "mul",
            "v212",
            "v213"
           ]
          ]
         ],
         [
          "mul",
          "v214",
          "v215"
         ]
        ]
       ],
       [
        "mul",
        "v216",
        "v217"
       ]
      ]
     ],
     [
      "mul",
      "v218",
      "v219"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v2687": [
        2,
        16,
        4
       ],
       "v2688": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v487": [
        2,
        16,
        1
       ],
       "v488": [
        2,
        1,
        4
       ],
       "v489": [
        2,
        16,
        1
       ],
       "v490": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v2687": [
        2,
        16,
        4
       ],
       "v2688": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v487": [
        2,
        16,
        1
       ],
       "v488": [
        2,
        1,
        4
       ],
       "v489": [
        2,
        16,
        1
       ],
       "v490": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v2687": [
        2,
        16,
        4
       ],
       "v2688": [
        2,
        16,
        4
       ],
       "v485": [
        2,
        16,
        4
       ],
       "v487": [
        2,
        16,
        1
       ],
       "v488": [
        2,
        1,
        4
       ],
       "v489": [
        2,
        16,
        1
       ],
       "v490": [
        2,
        1,
        4
       ]
      },
      {
       "v207": [
        2,
        16,
        1
       ],
       "v208": [
        2,
        16,
        1
       ],
       "v209": [
        2,
        16,
        1
       ],
       "v210": [
        2,
        16,
        1
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        16
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        16
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        16
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        16
       ],
       "v2687": [
        2,
        16,
        1
       ],
       "v2688": [
        2,
        16,
        16
       ],
       "v485": [
        2,
        16,
        1
       ],
       "v487": [
        2,
        16,
        1
       ],
       "v488": [
        2,
        1,
        16
       ],
       "v489": [
        2,
        16,
        1
       ],
       "v490": [
        2,
        1,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_c79a6fe687": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))), (mul 'v13', 'v14'))), (mul 'v15', 'v16'))), (mul 'v17', 'v18'))",
  "alpha_fold": "(synop2_d46808a970 'v0', 'v1', 'v2', 'v3', (synop_bb8c54c201 'v4', (synop_bb8c54c201 'v5', 'v6', 'v7', 'v8'), 'v9', 'v10'), 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18')",
  "embeds": [
   "synop3_304457cd8d",
   "synop3_4cdc06b88f"
  ],
  "embeds_shipped": [
   "synop2_d46808a970",
   "synop_bb8c54c201"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_c79a6fe687 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v696"
     },
     {
      "mvar": "v969"
     },
     {
      "mvar": "v970"
     },
     {
      "mvar": "v971"
     },
     {
      "mvar": "v972"
     },
     {
      "mvar": "v698"
     },
     {
      "mvar": "v699"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_c79a6fe687"
   },
   "missing_hooks": [],
   "name": "synop3_c79a6fe687.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v220"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v221"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v222"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v223"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v696"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v969"
                           },
                           {
                            "mvar": "v970"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v971"
                           },
                           {
                            "mvar": "v972"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v698"
                       },
                       {
                        "mvar": "v699"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v225"
                   },
                   {
                    "mvar": "v226"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v227"
               },
               {
                "mvar": "v228"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v229"
           },
           {
            "mvar": "v230"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v231"
       },
       {
        "mvar": "v232"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "args": [
       {
        "mvar": "v696"
       },
       {
        "args": [
         {
          "mvar": "v969"
         },
         {
          "mvar": "v970"
         },
         {
          "mvar": "v971"
         },
         {
          "mvar": "v972"
         }
        ],
        "attrs": {},
        "op": "synop_bb8c54c201"
       },
       {
        "mvar": "v698"
       },
       {
        "mvar": "v699"
       }
      ],
      "attrs": {},
      "op": "synop_bb8c54c201"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop2_d46808a970"
   },
   "missing_hooks": [],
   "name": "prim:synop3_c79a6fe687",
   "rhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v696"
     },
     {
      "mvar": "v969"
     },
     {
      "mvar": "v970"
     },
     {
      "mvar": "v971"
     },
     {
      "mvar": "v972"
     },
     {
      "mvar": "v698"
     },
     {
      "mvar": "v699"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_c79a6fe687"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v220",
     "v221",
     "v222",
     "v223",
     "v696",
     "v969",
     "v970",
     "v971",
     "v972",
     "v698",
     "v699",
     "v225",
     "v226",
     "v227",
     "v228",
     "v229",
     "v230",
     "v231",
     "v232"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v220",
       [
        "add",
        [
         "mul",
         "v221",
         [
          "add",
          [
           "mul",
           "v222",
           [
            "add",
            [
             "mul",
             "v223",
             [
              "add",
              [
               "mul",
               "v696",
               [
                "add",
                [
                 "mul",
                 "v969",
                 "v970"
                ],
                [
                 "mul",
                 "v971",
                 "v972"
                ]
               ]
              ],
              [
               "mul",
               "v698",
               "v699"
              ]
             ]
            ],
            [
             "mul",
             "v225",
             "v226"
            ]
           ]
          ],
          [
           "mul",
           "v227",
           "v228"
          ]
         ]
        ],
        [
         "mul",
         "v229",
         "v230"
        ]
       ]
      ],
      [
       "mul",
       "v231",
       "v232"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v220",
      [
       "add",
       [
        "mul",
        "v221",
        [
         "add",
         [
          "mul",
          "v222",
          [
           "add",
           [
            "mul",
            "v223",
            [
             "add",
             [
              "mul",
              "v696",
              [
               "add",
               [
                "mul",
                "v969",
                "v970"
               ],
               [
                "mul",
                "v971",
                "v972"
               ]
              ]
             ],
             [
              "mul",
              "v698",
              "v699"
             ]
            ]
           ],
           [
            "mul",
            "v225",
            "v226"
           ]
          ]
         ],
         [
          "mul",
          "v227",
          "v228"
         ]
        ]
       ],
       [
        "mul",
        "v229",
        "v230"
       ]
      ]
     ],
     [
      "mul",
      "v231",
      "v232"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v696": [
        2,
        16,
        4
       ],
       "v698": [
        2,
        16,
        1
       ],
       "v699": [
        2,
        1,
        4
       ],
       "v969": [
        2,
        16,
        4
       ],
       "v970": [
        2,
        16,
        4
       ],
       "v971": [
        2,
        16,
        1
       ],
       "v972": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v696": [
        2,
        16,
        4
       ],
       "v698": [
        2,
        16,
        1
       ],
       "v699": [
        2,
        1,
        4
       ],
       "v969": [
        2,
        16,
        4
       ],
       "v970": [
        2,
        16,
        4
       ],
       "v971": [
        2,
        16,
        1
       ],
       "v972": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v696": [
        2,
        16,
        4
       ],
       "v698": [
        2,
        16,
        1
       ],
       "v699": [
        2,
        1,
        4
       ],
       "v969": [
        2,
        16,
        4
       ],
       "v970": [
        2,
        16,
        4
       ],
       "v971": [
        2,
        16,
        1
       ],
       "v972": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        1
       ],
       "v221": [
        2,
        16,
        1
       ],
       "v222": [
        2,
        16,
        1
       ],
       "v223": [
        2,
        16,
        1
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        16
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        16
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        16
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        16
       ],
       "v696": [
        2,
        16,
        1
       ],
       "v698": [
        2,
        16,
        1
       ],
       "v699": [
        2,
        1,
        16
       ],
       "v969": [
        2,
        16,
        1
       ],
       "v970": [
        2,
        16,
        16
       ],
       "v971": [
        2,
        16,
        1
       ],
       "v972": [
        2,
        1,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_d5452c1c56": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), (mul 'v7', 'v8'))), 'v9')), (mul 'v10', 'v11'))), (mul 'v12', 'v13'))), (mul 'v14', 'v15'))), (mul 'v16', 'v17'))",
  "alpha_fold": "(synop2_d46808a970 'v0', 'v1', 'v2', 'v3', (add (mul 'v4', (synop_bb8c54c201 'v5', 'v6', 'v7', 'v8')), 'v9'), 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17')",
  "embeds": [],
  "embeds_shipped": [
   "synop2_d46808a970",
   "synop_bb8c54c201"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_d5452c1c56 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v2183"
     },
     {
      "mvar": "v2329"
     },
     {
      "mvar": "v2330"
     },
     {
      "mvar": "v2331"
     },
     {
      "mvar": "v2332"
     },
     {
      "mvar": "v600"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_d5452c1c56"
   },
   "missing_hooks": [],
   "name": "synop3_d5452c1c56.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v220"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v221"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v222"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v223"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v2183"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v2329"
                           },
                           {
                            "mvar": "v2330"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v2331"
                           },
                           {
                            "mvar": "v2332"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "mvar": "v600"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v225"
                   },
                   {
                    "mvar": "v226"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v227"
               },
               {
                "mvar": "v228"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v229"
           },
           {
            "mvar": "v230"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v231"
       },
       {
        "mvar": "v232"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v2183"
         },
         {
          "args": [
           {
            "mvar": "v2329"
           },
           {
            "mvar": "v2330"
           },
           {
            "mvar": "v2331"
           },
           {
            "mvar": "v2332"
           }
          ],
          "attrs": {},
          "op": "synop_bb8c54c201"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "mvar": "v600"
       }
      ],
      "attrs": {},
      "op": "add"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop2_d46808a970"
   },
   "missing_hooks": [],
   "name": "prim:synop3_d5452c1c56",
   "rhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v2183"
     },
     {
      "mvar": "v2329"
     },
     {
      "mvar": "v2330"
     },
     {
      "mvar": "v2331"
     },
     {
      "mvar": "v2332"
     },
     {
      "mvar": "v600"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_d5452c1c56"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v220",
     "v221",
     "v222",
     "v223",
     "v2183",
     "v2329",
     "v2330",
     "v2331",
     "v2332",
     "v600",
     "v225",
     "v226",
     "v227",
     "v228",
     "v229",
     "v230",
     "v231",
     "v232"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v220",
       [
        "add",
        [
         "mul",
         "v221",
         [
          "add",
          [
           "mul",
           "v222",
           [
            "add",
            [
             "mul",
             "v223",
             [
              "add",
              [
               "mul",
               "v2183",
               [
                "add",
                [
                 "mul",
                 "v2329",
                 "v2330"
                ],
                [
                 "mul",
                 "v2331",
                 "v2332"
                ]
               ]
              ],
              "v600"
             ]
            ],
            [
             "mul",
             "v225",
             "v226"
            ]
           ]
          ],
          [
           "mul",
           "v227",
           "v228"
          ]
         ]
        ],
        [
         "mul",
         "v229",
         "v230"
        ]
       ]
      ],
      [
       "mul",
       "v231",
       "v232"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v220",
      [
       "add",
       [
        "mul",
        "v221",
        [
         "add",
         [
          "mul",
          "v222",
          [
           "add",
           [
            "mul",
            "v223",
            [
             "add",
             [
              "mul",
              "v2183",
              [
               "add",
               [
                "mul",
                "v2329",
                "v2330"
               ],
               [
                "mul",
                "v2331",
                "v2332"
               ]
              ]
             ],
             "v600"
            ]
           ],
           [
            "mul",
            "v225",
            "v226"
           ]
          ]
         ],
         [
          "mul",
          "v227",
          "v228"
         ]
        ]
       ],
       [
        "mul",
        "v229",
        "v230"
       ]
      ]
     ],
     [
      "mul",
      "v231",
      "v232"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v2183": [
        2,
        16,
        4
       ],
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v2329": [
        2,
        16,
        4
       ],
       "v2330": [
        2,
        16,
        4
       ],
       "v2331": [
        2,
        16,
        1
       ],
       "v2332": [
        2,
        1,
        4
       ],
       "v600": [
        2,
        16,
        4
       ]
      },
      {
       "v2183": [
        2,
        16,
        4
       ],
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v2329": [
        2,
        16,
        4
       ],
       "v2330": [
        2,
        16,
        4
       ],
       "v2331": [
        2,
        16,
        1
       ],
       "v2332": [
        2,
        1,
        4
       ],
       "v600": [
        2,
        16,
        4
       ]
      },
      {
       "v2183": [
        2,
        16,
        4
       ],
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v2329": [
        2,
        16,
        4
       ],
       "v2330": [
        2,
        16,
        4
       ],
       "v2331": [
        2,
        16,
        1
       ],
       "v2332": [
        2,
        1,
        4
       ],
       "v600": [
        2,
        16,
        4
       ]
      },
      {
       "v2183": [
        2,
        16,
        1
       ],
       "v220": [
        2,
        16,
        1
       ],
       "v221": [
        2,
        16,
        1
       ],
       "v222": [
        2,
        16,
        1
       ],
       "v223": [
        2,
        16,
        1
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        16
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        16
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        16
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        16
       ],
       "v2329": [
        2,
        16,
        1
       ],
       "v2330": [
        2,
        16,
        16
       ],
       "v2331": [
        2,
        16,
        1
       ],
       "v2332": [
        2,
        1,
        16
       ],
       "v600": [
        2,
        16,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_d8a399f647": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))), (mul 'v13', 'v14'))), (mul 'v15', 'v16'))), (mul 'v17', 'v18'))",
  "alpha_fold": "(synop2_d46808a970 'v0', 'v1', 'v2', 'v3', (synop_f3290a1163 'v4', 'v5', 'v6', (mul 'v7', 'v8'), 'v9', 'v10'), 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18')",
  "embeds": [
   "synop3_304457cd8d",
   "synop3_4cdc06b88f"
  ],
  "embeds_shipped": [
   "synop2_d46808a970",
   "synop_f3290a1163"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_d8a399f647 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v713"
     },
     {
      "mvar": "v714"
     },
     {
      "mvar": "v715"
     },
     {
      "mvar": "v3718"
     },
     {
      "mvar": "v3719"
     },
     {
      "mvar": "v717"
     },
     {
      "mvar": "v718"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_d8a399f647"
   },
   "missing_hooks": [],
   "name": "synop3_d8a399f647.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v220"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v221"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v222"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v223"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v713"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v714"
                           },
                           {
                            "mvar": "v715"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v3718"
                           },
                           {
                            "mvar": "v3719"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v717"
                       },
                       {
                        "mvar": "v718"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v225"
                   },
                   {
                    "mvar": "v226"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v227"
               },
               {
                "mvar": "v228"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v229"
           },
           {
            "mvar": "v230"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v231"
       },
       {
        "mvar": "v232"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "args": [
       {
        "mvar": "v713"
       },
       {
        "mvar": "v714"
       },
       {
        "mvar": "v715"
       },
       {
        "args": [
         {
          "mvar": "v3718"
         },
         {
          "mvar": "v3719"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "mvar": "v717"
       },
       {
        "mvar": "v718"
       }
      ],
      "attrs": {},
      "op": "synop_f3290a1163"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop2_d46808a970"
   },
   "missing_hooks": [],
   "name": "prim:synop3_d8a399f647",
   "rhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v713"
     },
     {
      "mvar": "v714"
     },
     {
      "mvar": "v715"
     },
     {
      "mvar": "v3718"
     },
     {
      "mvar": "v3719"
     },
     {
      "mvar": "v717"
     },
     {
      "mvar": "v718"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_d8a399f647"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v220",
     "v221",
     "v222",
     "v223",
     "v713",
     "v714",
     "v715",
     "v3718",
     "v3719",
     "v717",
     "v718",
     "v225",
     "v226",
     "v227",
     "v228",
     "v229",
     "v230",
     "v231",
     "v232"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v220",
       [
        "add",
        [
         "mul",
         "v221",
         [
          "add",
          [
           "mul",
           "v222",
           [
            "add",
            [
             "mul",
             "v223",
             [
              "add",
              [
               "mul",
               "v713",
               [
                "add",
                [
                 "mul",
                 "v714",
                 "v715"
                ],
                [
                 "mul",
                 "v3718",
                 "v3719"
                ]
               ]
              ],
              [
               "mul",
               "v717",
               "v718"
              ]
             ]
            ],
            [
             "mul",
             "v225",
             "v226"
            ]
           ]
          ],
          [
           "mul",
           "v227",
           "v228"
          ]
         ]
        ],
        [
         "mul",
         "v229",
         "v230"
        ]
       ]
      ],
      [
       "mul",
       "v231",
       "v232"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v220",
      [
       "add",
       [
        "mul",
        "v221",
        [
         "add",
         [
          "mul",
          "v222",
          [
           "add",
           [
            "mul",
            "v223",
            [
             "add",
             [
              "mul",
              "v713",
              [
               "add",
               [
                "mul",
                "v714",
                "v715"
               ],
               [
                "mul",
                "v3718",
                "v3719"
               ]
              ]
             ],
             [
              "mul",
              "v717",
              "v718"
             ]
            ]
           ],
           [
            "mul",
            "v225",
            "v226"
           ]
          ]
         ],
         [
          "mul",
          "v227",
          "v228"
         ]
        ]
       ],
       [
        "mul",
        "v229",
        "v230"
       ]
      ]
     ],
     [
      "mul",
      "v231",
      "v232"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v3718": [
        2,
        16,
        1
       ],
       "v3719": [
        2,
        1,
        4
       ],
       "v713": [
        2,
        16,
        4
       ],
       "v714": [
        2,
        16,
        4
       ],
       "v715": [
        2,
        16,
        4
       ],
       "v717": [
        2,
        16,
        1
       ],
       "v718": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v3718": [
        2,
        16,
        1
       ],
       "v3719": [
        2,
        1,
        4
       ],
       "v713": [
        2,
        16,
        4
       ],
       "v714": [
        2,
        16,
        4
       ],
       "v715": [
        2,
        16,
        4
       ],
       "v717": [
        2,
        16,
        1
       ],
       "v718": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v3718": [
        2,
        16,
        1
       ],
       "v3719": [
        2,
        1,
        4
       ],
       "v713": [
        2,
        16,
        4
       ],
       "v714": [
        2,
        16,
        4
       ],
       "v715": [
        2,
        16,
        4
       ],
       "v717": [
        2,
        16,
        1
       ],
       "v718": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        1
       ],
       "v221": [
        2,
        16,
        1
       ],
       "v222": [
        2,
        16,
        1
       ],
       "v223": [
        2,
        16,
        1
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        16
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        16
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        16
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        16
       ],
       "v3718": [
        2,
        16,
        1
       ],
       "v3719": [
        2,
        1,
        16
       ],
       "v713": [
        2,
        16,
        1
       ],
       "v714": [
        2,
        16,
        1
       ],
       "v715": [
        2,
        16,
        16
       ],
       "v717": [
        2,
        16,
        1
       ],
       "v718": [
        2,
        1,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_e8abc00db0": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), (mul 'v7', 'v8'))), 'v9')), (mul 'v10', 'v11'))), (mul 'v12', 'v13'))), (mul 'v14', 'v15'))), (mul 'v16', 'v17'))",
  "alpha_fold": "(synop2_d46808a970 'v0', 'v1', 'v2', 'v3', (synop_6c709f774a 'v4', 'v5', 'v6', 'v7', 'v8', 'v9'), 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17')",
  "embeds": [],
  "embeds_shipped": [
   "synop2_d46808a970",
   "synop_6c709f774a"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_e8abc00db0 unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v678"
     },
     {
      "mvar": "v679"
     },
     {
      "mvar": "v680"
     },
     {
      "mvar": "v681"
     },
     {
      "mvar": "v682"
     },
     {
      "mvar": "v683"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_e8abc00db0"
   },
   "missing_hooks": [],
   "name": "synop3_e8abc00db0.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v220"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v221"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v222"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v223"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v678"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v679"
                           },
                           {
                            "mvar": "v680"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v681"
                           },
                           {
                            "mvar": "v682"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "mvar": "v683"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v225"
                   },
                   {
                    "mvar": "v226"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v227"
               },
               {
                "mvar": "v228"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v229"
           },
           {
            "mvar": "v230"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v231"
       },
       {
        "mvar": "v232"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "args": [
       {
        "mvar": "v678"
       },
       {
        "mvar": "v679"
       },
       {
        "mvar": "v680"
       },
       {
        "mvar": "v681"
       },
       {
        "mvar": "v682"
       },
       {
        "mvar": "v683"
       }
      ],
      "attrs": {},
      "op": "synop_6c709f774a"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop2_d46808a970"
   },
   "missing_hooks": [],
   "name": "prim:synop3_e8abc00db0",
   "rhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v678"
     },
     {
      "mvar": "v679"
     },
     {
      "mvar": "v680"
     },
     {
      "mvar": "v681"
     },
     {
      "mvar": "v682"
     },
     {
      "mvar": "v683"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_e8abc00db0"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v220",
     "v221",
     "v222",
     "v223",
     "v678",
     "v679",
     "v680",
     "v681",
     "v682",
     "v683",
     "v225",
     "v226",
     "v227",
     "v228",
     "v229",
     "v230",
     "v231",
     "v232"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v220",
       [
        "add",
        [
         "mul",
         "v221",
         [
          "add",
          [
           "mul",
           "v222",
           [
            "add",
            [
             "mul",
             "v223",
             [
              "add",
              [
               "mul",
               "v678",
               [
                "add",
                [
                 "mul",
                 "v679",
                 "v680"
                ],
                [
                 "mul",
                 "v681",
                 "v682"
                ]
               ]
              ],
              "v683"
             ]
            ],
            [
             "mul",
             "v225",
             "v226"
            ]
           ]
          ],
          [
           "mul",
           "v227",
           "v228"
          ]
         ]
        ],
        [
         "mul",
         "v229",
         "v230"
        ]
       ]
      ],
      [
       "mul",
       "v231",
       "v232"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v220",
      [
       "add",
       [
        "mul",
        "v221",
        [
         "add",
         [
          "mul",
          "v222",
          [
           "add",
           [
            "mul",
            "v223",
            [
             "add",
             [
              "mul",
              "v678",
              [
               "add",
               [
                "mul",
                "v679",
                "v680"
               ],
               [
                "mul",
                "v681",
                "v682"
               ]
              ]
             ],
             "v683"
            ]
           ],
           [
            "mul",
            "v225",
            "v226"
           ]
          ]
         ],
         [
          "mul",
          "v227",
          "v228"
         ]
        ]
       ],
       [
        "mul",
        "v229",
        "v230"
       ]
      ]
     ],
     [
      "mul",
      "v231",
      "v232"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v678": [
        2,
        16,
        4
       ],
       "v679": [
        2,
        16,
        4
       ],
       "v680": [
        2,
        16,
        4
       ],
       "v681": [
        2,
        16,
        1
       ],
       "v682": [
        2,
        1,
        4
       ],
       "v683": [
        2,
        16,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v678": [
        2,
        16,
        4
       ],
       "v679": [
        2,
        16,
        4
       ],
       "v680": [
        2,
        16,
        4
       ],
       "v681": [
        2,
        16,
        1
       ],
       "v682": [
        2,
        1,
        4
       ],
       "v683": [
        2,
        16,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v678": [
        2,
        16,
        4
       ],
       "v679": [
        2,
        16,
        4
       ],
       "v680": [
        2,
        16,
        4
       ],
       "v681": [
        2,
        16,
        1
       ],
       "v682": [
        2,
        1,
        4
       ],
       "v683": [
        2,
        16,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        1
       ],
       "v221": [
        2,
        16,
        1
       ],
       "v222": [
        2,
        16,
        1
       ],
       "v223": [
        2,
        16,
        1
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        16
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        16
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        16
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        16
       ],
       "v678": [
        2,
        16,
        1
       ],
       "v679": [
        2,
        16,
        1
       ],
       "v680": [
        2,
        16,
        16
       ],
       "v681": [
        2,
        16,
        1
       ],
       "v682": [
        2,
        1,
        16
       ],
       "v683": [
        2,
        16,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_ed5918edde": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add 'v5', (mul 'v6', 'v7'))), (mul 'v8', 'v9'))), (mul 'v10', 'v11'))), (mul 'v12', 'v13'))), (mul 'v14', 'v15'))), (mul 'v16', 'v17'))",
  "alpha_fold": "(synop2_d46808a970 'v0', 'v1', 'v2', 'v3', (synop_2b8cfd1c90 'v4', 'v5', 'v6', 'v7', 'v8', 'v9'), 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17')",
  "embeds": [
   "synop3_39dbcda988"
  ],
  "embeds_shipped": [
   "synop2_d46808a970",
   "synop_2b8cfd1c90"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_ed5918edde unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v633"
     },
     {
      "mvar": "v634"
     },
     {
      "mvar": "v635"
     },
     {
      "mvar": "v636"
     },
     {
      "mvar": "v637"
     },
     {
      "mvar": "v638"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_ed5918edde"
   },
   "missing_hooks": [],
   "name": "synop3_ed5918edde.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v220"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v221"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v222"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v223"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v633"
                       },
                       {
                        "args": [
                         {
                          "mvar": "v634"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v635"
                           },
                           {
                            "mvar": "v636"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v637"
                       },
                       {
                        "mvar": "v638"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v225"
                   },
                   {
                    "mvar": "v226"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v227"
               },
               {
                "mvar": "v228"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v229"
           },
           {
            "mvar": "v230"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v231"
       },
       {
        "mvar": "v232"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "args": [
       {
        "mvar": "v633"
       },
       {
        "mvar": "v634"
       },
       {
        "mvar": "v635"
       },
       {
        "mvar": "v636"
       },
       {
        "mvar": "v637"
       },
       {
        "mvar": "v638"
       }
      ],
      "attrs": {},
      "op": "synop_2b8cfd1c90"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop2_d46808a970"
   },
   "missing_hooks": [],
   "name": "prim:synop3_ed5918edde",
   "rhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v633"
     },
     {
      "mvar": "v634"
     },
     {
      "mvar": "v635"
     },
     {
      "mvar": "v636"
     },
     {
      "mvar": "v637"
     },
     {
      "mvar": "v638"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_ed5918edde"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v220",
     "v221",
     "v222",
     "v223",
     "v633",
     "v634",
     "v635",
     "v636",
     "v637",
     "v638",
     "v225",
     "v226",
     "v227",
     "v228",
     "v229",
     "v230",
     "v231",
     "v232"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v220",
       [
        "add",
        [
         "mul",
         "v221",
         [
          "add",
          [
           "mul",
           "v222",
           [
            "add",
            [
             "mul",
             "v223",
             [
              "add",
              [
               "mul",
               "v633",
               [
                "add",
                "v634",
                [
                 "mul",
                 "v635",
                 "v636"
                ]
               ]
              ],
              [
               "mul",
               "v637",
               "v638"
              ]
             ]
            ],
            [
             "mul",
             "v225",
             "v226"
            ]
           ]
          ],
          [
           "mul",
           "v227",
           "v228"
          ]
         ]
        ],
        [
         "mul",
         "v229",
         "v230"
        ]
       ]
      ],
      [
       "mul",
       "v231",
       "v232"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v220",
      [
       "add",
       [
        "mul",
        "v221",
        [
         "add",
         [
          "mul",
          "v222",
          [
           "add",
           [
            "mul",
            "v223",
            [
             "add",
             [
              "mul",
              "v633",
              [
               "add",
               "v634",
               [
                "mul",
                "v635",
                "v636"
               ]
              ]
             ],
             [
              "mul",
              "v637",
              "v638"
             ]
            ]
           ],
           [
            "mul",
            "v225",
            "v226"
           ]
          ]
         ],
         [
          "mul",
          "v227",
          "v228"
         ]
        ]
       ],
       [
        "mul",
        "v229",
        "v230"
       ]
      ]
     ],
     [
      "mul",
      "v231",
      "v232"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v633": [
        2,
        16,
        4
       ],
       "v634": [
        2,
        16,
        4
       ],
       "v635": [
        2,
        16,
        1
       ],
       "v636": [
        2,
        1,
        4
       ],
       "v637": [
        2,
        16,
        1
       ],
       "v638": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v633": [
        2,
        16,
        4
       ],
       "v634": [
        2,
        16,
        4
       ],
       "v635": [
        2,
        16,
        1
       ],
       "v636": [
        2,
        1,
        4
       ],
       "v637": [
        2,
        16,
        1
       ],
       "v638": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v633": [
        2,
        16,
        4
       ],
       "v634": [
        2,
        16,
        4
       ],
       "v635": [
        2,
        16,
        1
       ],
       "v636": [
        2,
        1,
        4
       ],
       "v637": [
        2,
        16,
        1
       ],
       "v638": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        1
       ],
       "v221": [
        2,
        16,
        1
       ],
       "v222": [
        2,
        16,
        1
       ],
       "v223": [
        2,
        16,
        1
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        16
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        16
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        16
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        16
       ],
       "v633": [
        2,
        16,
        1
       ],
       "v634": [
        2,
        16,
        16
       ],
       "v635": [
        2,
        16,
        1
       ],
       "v636": [
        2,
        1,
        16
       ],
       "v637": [
        2,
        16,
        1
       ],
       "v638": [
        2,
        1,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_efb0c0d7ce": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', (add (mul 'v5', 'v6'), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))), (mul 'v13', 'v14'))), (mul 'v15', 'v16'))), (mul 'v17', 'v18'))",
  "alpha_fold": "(synop2_d46808a970 'v0', 'v1', 'v2', 'v3', (synop_6c709f774a 'v4', 'v5', 'v6', 'v7', 'v8', (mul 'v9', 'v10')), 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18')",
  "embeds": [
   "synop3_304457cd8d",
   "synop3_4cdc06b88f"
  ],
  "embeds_shipped": [
   "synop2_d46808a970",
   "synop_6c709f774a"
  ],
  "equal_sites": 9,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_efb0c0d7ce unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v678"
     },
     {
      "mvar": "v679"
     },
     {
      "mvar": "v680"
     },
     {
      "mvar": "v681"
     },
     {
      "mvar": "v682"
     },
     {
      "mvar": "v3522"
     },
     {
      "mvar": "v3523"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_efb0c0d7ce"
   },
   "missing_hooks": [],
   "name": "synop3_efb0c0d7ce.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v220"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v221"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v222"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v223"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v678"
                       },
                       {
                        "args": [
                         {
                          "args": [
                           {
                            "mvar": "v679"
                           },
                           {
                            "mvar": "v680"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         },
                         {
                          "args": [
                           {
                            "mvar": "v681"
                           },
                           {
                            "mvar": "v682"
                           }
                          ],
                          "attrs": {},
                          "op": "mul"
                         }
                        ],
                        "attrs": {},
                        "op": "add"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "args": [
                       {
                        "mvar": "v3522"
                       },
                       {
                        "mvar": "v3523"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v225"
                   },
                   {
                    "mvar": "v226"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v227"
               },
               {
                "mvar": "v228"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v229"
           },
           {
            "mvar": "v230"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v231"
       },
       {
        "mvar": "v232"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "args": [
       {
        "mvar": "v678"
       },
       {
        "mvar": "v679"
       },
       {
        "mvar": "v680"
       },
       {
        "mvar": "v681"
       },
       {
        "mvar": "v682"
       },
       {
        "args": [
         {
          "mvar": "v3522"
         },
         {
          "mvar": "v3523"
         }
        ],
        "attrs": {},
        "op": "mul"
       }
      ],
      "attrs": {},
      "op": "synop_6c709f774a"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop2_d46808a970"
   },
   "missing_hooks": [],
   "name": "prim:synop3_efb0c0d7ce",
   "rhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v678"
     },
     {
      "mvar": "v679"
     },
     {
      "mvar": "v680"
     },
     {
      "mvar": "v681"
     },
     {
      "mvar": "v682"
     },
     {
      "mvar": "v3522"
     },
     {
      "mvar": "v3523"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_efb0c0d7ce"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v220",
     "v221",
     "v222",
     "v223",
     "v678",
     "v679",
     "v680",
     "v681",
     "v682",
     "v3522",
     "v3523",
     "v225",
     "v226",
     "v227",
     "v228",
     "v229",
     "v230",
     "v231",
     "v232"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v220",
       [
        "add",
        [
         "mul",
         "v221",
         [
          "add",
          [
           "mul",
           "v222",
           [
            "add",
            [
             "mul",
             "v223",
             [
              "add",
              [
               "mul",
               "v678",
               [
                "add",
                [
                 "mul",
                 "v679",
                 "v680"
                ],
                [
                 "mul",
                 "v681",
                 "v682"
                ]
               ]
              ],
              [
               "mul",
               "v3522",
               "v3523"
              ]
             ]
            ],
            [
             "mul",
             "v225",
             "v226"
            ]
           ]
          ],
          [
           "mul",
           "v227",
           "v228"
          ]
         ]
        ],
        [
         "mul",
         "v229",
         "v230"
        ]
       ]
      ],
      [
       "mul",
       "v231",
       "v232"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v220",
      [
       "add",
       [
        "mul",
        "v221",
        [
         "add",
         [
          "mul",
          "v222",
          [
           "add",
           [
            "mul",
            "v223",
            [
             "add",
             [
              "mul",
              "v678",
              [
               "add",
               [
                "mul",
                "v679",
                "v680"
               ],
               [
                "mul",
                "v681",
                "v682"
               ]
              ]
             ],
             [
              "mul",
              "v3522",
              "v3523"
             ]
            ]
           ],
           [
            "mul",
            "v225",
            "v226"
           ]
          ]
         ],
         [
          "mul",
          "v227",
          "v228"
         ]
        ]
       ],
       [
        "mul",
        "v229",
        "v230"
       ]
      ]
     ],
     [
      "mul",
      "v231",
      "v232"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v3522": [
        2,
        16,
        1
       ],
       "v3523": [
        2,
        1,
        4
       ],
       "v678": [
        2,
        16,
        4
       ],
       "v679": [
        2,
        16,
        4
       ],
       "v680": [
        2,
        16,
        4
       ],
       "v681": [
        2,
        16,
        1
       ],
       "v682": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v3522": [
        2,
        16,
        1
       ],
       "v3523": [
        2,
        1,
        4
       ],
       "v678": [
        2,
        16,
        4
       ],
       "v679": [
        2,
        16,
        4
       ],
       "v680": [
        2,
        16,
        4
       ],
       "v681": [
        2,
        16,
        1
       ],
       "v682": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v3522": [
        2,
        16,
        1
       ],
       "v3523": [
        2,
        1,
        4
       ],
       "v678": [
        2,
        16,
        4
       ],
       "v679": [
        2,
        16,
        4
       ],
       "v680": [
        2,
        16,
        4
       ],
       "v681": [
        2,
        16,
        1
       ],
       "v682": [
        2,
        1,
        4
       ]
      },
      {
       "v220": [
        2,
        16,
        1
       ],
       "v221": [
        2,
        16,
        1
       ],
       "v222": [
        2,
        16,
        1
       ],
       "v223": [
        2,
        16,
        1
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        16
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        16
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        16
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        16
       ],
       "v3522": [
        2,
        16,
        1
       ],
       "v3523": [
        2,
        1,
        16
       ],
       "v678": [
        2,
        16,
        1
       ],
       "v679": [
        2,
        16,
        1
       ],
       "v680": [
        2,
        16,
        16
       ],
       "v681": [
        2,
        16,
        1
       ],
       "v682": [
        2,
        1,
        16
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 9
 },
 "synop3_f28f7ffd1a": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', 'v5'), 'v6')), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))), (mul 'v13', 'v14'))",
  "alpha_fold": "(synop2_35bbd03301 'v0', 'v1', 'v2', 'v3', (add (mul 'v4', 'v5'), 'v6'), 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14')",
  "embeds": [],
  "embeds_shipped": [
   "synop2_35bbd03301"
  ],
  "equal_sites": 12,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_f28f7ffd1a unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v1681"
     },
     {
      "mvar": "v1682"
     },
     {
      "mvar": "v452"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_f28f7ffd1a"
   },
   "missing_hooks": [],
   "name": "synop3_f28f7ffd1a.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v207"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v208"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v209"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v210"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v1681"
                       },
                       {
                        "mvar": "v1682"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "mvar": "v452"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v212"
                   },
                   {
                    "mvar": "v213"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v214"
               },
               {
                "mvar": "v215"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v216"
           },
           {
            "mvar": "v217"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v218"
       },
       {
        "mvar": "v219"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v1681"
         },
         {
          "mvar": "v1682"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "mvar": "v452"
       }
      ],
      "attrs": {},
      "op": "add"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop2_35bbd03301"
   },
   "missing_hooks": [],
   "name": "prim:synop3_f28f7ffd1a",
   "rhs": {
    "args": [
     {
      "mvar": "v207"
     },
     {
      "mvar": "v208"
     },
     {
      "mvar": "v209"
     },
     {
      "mvar": "v210"
     },
     {
      "mvar": "v1681"
     },
     {
      "mvar": "v1682"
     },
     {
      "mvar": "v452"
     },
     {
      "mvar": "v212"
     },
     {
      "mvar": "v213"
     },
     {
      "mvar": "v214"
     },
     {
      "mvar": "v215"
     },
     {
      "mvar": "v216"
     },
     {
      "mvar": "v217"
     },
     {
      "mvar": "v218"
     },
     {
      "mvar": "v219"
     }
    ],
    "attrs": {},
    "op": "synop3_f28f7ffd1a"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v207",
     "v208",
     "v209",
     "v210",
     "v1681",
     "v1682",
     "v452",
     "v212",
     "v213",
     "v214",
     "v215",
     "v216",
     "v217",
     "v218",
     "v219"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v207",
       [
        "add",
        [
         "mul",
         "v208",
         [
          "add",
          [
           "mul",
           "v209",
           [
            "add",
            [
             "mul",
             "v210",
             [
              "add",
              [
               "mul",
               "v1681",
               "v1682"
              ],
              "v452"
             ]
            ],
            [
             "mul",
             "v212",
             "v213"
            ]
           ]
          ],
          [
           "mul",
           "v214",
           "v215"
          ]
         ]
        ],
        [
         "mul",
         "v216",
         "v217"
        ]
       ]
      ],
      [
       "mul",
       "v218",
       "v219"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v207",
      [
       "add",
       [
        "mul",
        "v208",
        [
         "add",
         [
          "mul",
          "v209",
          [
           "add",
           [
            "mul",
            "v210",
            [
             "add",
             [
              "mul",
              "v1681",
              "v1682"
             ],
             "v452"
            ]
           ],
           [
            "mul",
            "v212",
            "v213"
           ]
          ]
         ],
         [
          "mul",
          "v214",
          "v215"
         ]
        ]
       ],
       [
        "mul",
        "v216",
        "v217"
       ]
      ]
     ],
     [
      "mul",
      "v218",
      "v219"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v1681": [
        2,
        16,
        4
       ],
       "v1682": [
        2,
        16,
        4
       ],
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v452": [
        2,
        16,
        4
       ]
      },
      {
       "v1681": [
        2,
        16,
        4
       ],
       "v1682": [
        2,
        16,
        4
       ],
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v452": [
        2,
        16,
        4
       ]
      },
      {
       "v1681": [
        2,
        16,
        4
       ],
       "v1682": [
        2,
        16,
        4
       ],
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v452": [
        2,
        16,
        4
       ]
      },
      {
       "v1681": [
        2,
        16,
        4
       ],
       "v1682": [
        2,
        16,
        4
       ],
       "v207": [
        2,
        16,
        4
       ],
       "v208": [
        2,
        16,
        4
       ],
       "v209": [
        2,
        16,
        4
       ],
       "v210": [
        2,
        16,
        4
       ],
       "v212": [
        2,
        16,
        1
       ],
       "v213": [
        2,
        1,
        4
       ],
       "v214": [
        2,
        16,
        1
       ],
       "v215": [
        2,
        1,
        4
       ],
       "v216": [
        2,
        16,
        1
       ],
       "v217": [
        2,
        1,
        4
       ],
       "v218": [
        2,
        16,
        1
       ],
       "v219": [
        2,
        1,
        4
       ],
       "v452": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 12
 },
 "synop3_f9c72e0a8f": {
  "alpha": "(add (mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', 'v5'), 'v6')), (mul 'v7', 'v8'))), (mul 'v9', 'v10'))), (mul 'v11', 'v12'))), (mul 'v13', 'v14'))",
  "alpha_fold": "(synop2_d46808a970 'v0', 'v1', 'v2', 'v3', (add (mul 'v4', 'v5'), 'v6'), 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14')",
  "embeds": [],
  "embeds_shipped": [
   "synop2_d46808a970"
  ],
  "equal_sites": 12,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_f9c72e0a8f unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v2183"
     },
     {
      "mvar": "v2184"
     },
     {
      "mvar": "v600"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_f9c72e0a8f"
   },
   "missing_hooks": [],
   "name": "synop3_f9c72e0a8f.expand",
   "rhs": {
    "args": [
     {
      "args": [
       {
        "mvar": "v220"
       },
       {
        "args": [
         {
          "args": [
           {
            "mvar": "v221"
           },
           {
            "args": [
             {
              "args": [
               {
                "mvar": "v222"
               },
               {
                "args": [
                 {
                  "args": [
                   {
                    "mvar": "v223"
                   },
                   {
                    "args": [
                     {
                      "args": [
                       {
                        "mvar": "v2183"
                       },
                       {
                        "mvar": "v2184"
                       }
                      ],
                      "attrs": {},
                      "op": "mul"
                     },
                     {
                      "mvar": "v600"
                     }
                    ],
                    "attrs": {},
                    "op": "add"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 },
                 {
                  "args": [
                   {
                    "mvar": "v225"
                   },
                   {
                    "mvar": "v226"
                   }
                  ],
                  "attrs": {},
                  "op": "mul"
                 }
                ],
                "attrs": {},
                "op": "add"
               }
              ],
              "attrs": {},
              "op": "mul"
             },
             {
              "args": [
               {
                "mvar": "v227"
               },
               {
                "mvar": "v228"
               }
              ],
              "attrs": {},
              "op": "mul"
             }
            ],
            "attrs": {},
            "op": "add"
           }
          ],
          "attrs": {},
          "op": "mul"
         },
         {
          "args": [
           {
            "mvar": "v229"
           },
           {
            "mvar": "v230"
           }
          ],
          "attrs": {},
          "op": "mul"
         }
        ],
        "attrs": {},
        "op": "add"
       }
      ],
      "attrs": {},
      "op": "mul"
     },
     {
      "args": [
       {
        "mvar": "v231"
       },
       {
        "mvar": "v232"
       }
      ],
      "attrs": {},
      "op": "mul"
     }
    ],
    "attrs": {},
    "op": "add"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v2183"
         },
         {
          "mvar": "v2184"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "mvar": "v600"
       }
      ],
      "attrs": {},
      "op": "add"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop2_d46808a970"
   },
   "missing_hooks": [],
   "name": "prim:synop3_f9c72e0a8f",
   "rhs": {
    "args": [
     {
      "mvar": "v220"
     },
     {
      "mvar": "v221"
     },
     {
      "mvar": "v222"
     },
     {
      "mvar": "v223"
     },
     {
      "mvar": "v2183"
     },
     {
      "mvar": "v2184"
     },
     {
      "mvar": "v600"
     },
     {
      "mvar": "v225"
     },
     {
      "mvar": "v226"
     },
     {
      "mvar": "v227"
     },
     {
      "mvar": "v228"
     },
     {
      "mvar": "v229"
     },
     {
      "mvar": "v230"
     },
     {
      "mvar": "v231"
     },
     {
      "mvar": "v232"
     }
    ],
    "attrs": {},
    "op": "synop3_f9c72e0a8f"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v220",
     "v221",
     "v222",
     "v223",
     "v2183",
     "v2184",
     "v600",
     "v225",
     "v226",
     "v227",
     "v228",
     "v229",
     "v230",
     "v231",
     "v232"
    ],
    "handler_body": [
     "ew",
     [
      "add",
      [
       "mul",
       "v220",
       [
        "add",
        [
         "mul",
         "v221",
         [
          "add",
          [
           "mul",
           "v222",
           [
            "add",
            [
             "mul",
             "v223",
             [
              "add",
              [
               "mul",
               "v2183",
               "v2184"
              ],
              "v600"
             ]
            ],
            [
             "mul",
             "v225",
             "v226"
            ]
           ]
          ],
          [
           "mul",
           "v227",
           "v228"
          ]
         ]
        ],
        [
         "mul",
         "v229",
         "v230"
        ]
       ]
      ],
      [
       "mul",
       "v231",
       "v232"
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "add",
     [
      "mul",
      "v220",
      [
       "add",
       [
        "mul",
        "v221",
        [
         "add",
         [
          "mul",
          "v222",
          [
           "add",
           [
            "mul",
            "v223",
            [
             "add",
             [
              "mul",
              "v2183",
              "v2184"
             ],
             "v600"
            ]
           ],
           [
            "mul",
            "v225",
            "v226"
           ]
          ]
         ],
         [
          "mul",
          "v227",
          "v228"
         ]
        ]
       ],
       [
        "mul",
        "v229",
        "v230"
       ]
      ]
     ],
     [
      "mul",
      "v231",
      "v232"
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v2183": [
        2,
        16,
        4
       ],
       "v2184": [
        2,
        16,
        4
       ],
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v600": [
        2,
        16,
        4
       ]
      },
      {
       "v2183": [
        2,
        16,
        4
       ],
       "v2184": [
        2,
        16,
        4
       ],
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v600": [
        2,
        16,
        4
       ]
      },
      {
       "v2183": [
        2,
        16,
        4
       ],
       "v2184": [
        2,
        16,
        4
       ],
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v600": [
        2,
        16,
        4
       ]
      },
      {
       "v2183": [
        2,
        16,
        4
       ],
       "v2184": [
        2,
        16,
        4
       ],
       "v220": [
        2,
        16,
        4
       ],
       "v221": [
        2,
        16,
        4
       ],
       "v222": [
        2,
        16,
        4
       ],
       "v223": [
        2,
        16,
        4
       ],
       "v225": [
        2,
        16,
        1
       ],
       "v226": [
        2,
        1,
        4
       ],
       "v227": [
        2,
        16,
        1
       ],
       "v228": [
        2,
        1,
        4
       ],
       "v229": [
        2,
        16,
        1
       ],
       "v230": [
        2,
        1,
        4
       ],
       "v231": [
        2,
        16,
        1
       ],
       "v232": [
        2,
        1,
        4
       ],
       "v600": [
        2,
        16,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 12
 },
 "synop3_fbbba382fc": {
  "alpha": "(mul 'v0', (add (mul 'v1', (add (mul 'v2', (add (mul 'v3', (add (mul 'v4', 'v5'), (mul 'v6', 'v7'))), (mul 'v8', 'v9'))), (mul 'v10', 'v11'))), (mul 'v12', 'v13')))",
  "alpha_fold": "(mul 'v0', (synop2_35bbd03301 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13'))",
  "embeds": [],
  "embeds_shipped": [
   "synop2_35bbd03301"
  ],
  "equal_sites": 12,
  "expand": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "synop3_fbbba382fc unfolds to its mined denotation \u2014 the definitional premise the cert leg anchors on",
   "lhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "mvar": "v323"
     },
     {
      "mvar": "v324"
     },
     {
      "mvar": "v325"
     },
     {
      "mvar": "v326"
     },
     {
      "mvar": "v327"
     },
     {
      "mvar": "v328"
     },
     {
      "mvar": "v329"
     },
     {
      "mvar": "v330"
     },
     {
      "mvar": "v331"
     },
     {
      "mvar": "v332"
     },
     {
      "mvar": "v333"
     },
     {
      "mvar": "v334"
     },
     {
      "mvar": "v335"
     }
    ],
    "attrs": {},
    "op": "synop3_fbbba382fc"
   },
   "missing_hooks": [],
   "name": "synop3_fbbba382fc.expand",
   "rhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "args": [
       {
        "args": [
         {
          "mvar": "v323"
         },
         {
          "args": [
           {
            "args": [
             {
              "mvar": "v324"
             },
             {
              "args": [
               {
                "args": [
                 {
                  "mvar": "v325"
                 },
                 {
                  "args": [
                   {
                    "args": [
                     {
                      "mvar": "v326"
                     },
                     {
                      "mvar": "v327"
                     }
                    ],
                    "attrs": {},
                    "op": "mul"
                   },
                   {
                    "args": [
                     {
                      "mvar": "v328"
                     },
                     {
                      "mvar": "v329"
                     }
                    ],
                    "attrs": {},
                    "op": "mul"
                   }
                  ],
                  "attrs": {},
                  "op": "add"
                 }
                ],
                "attrs": {},
                "op": "mul"
               },
               {
                "args": [
                 {
                  "mvar": "v330"
                 },
                 {
                  "mvar": "v331"
                 }
                ],
                "attrs": {},
                "op": "mul"
               }
              ],
              "attrs": {},
              "op": "add"
             }
            ],
            "attrs": {},
            "op": "mul"
           },
           {
            "args": [
             {
              "mvar": "v332"
             },
             {
              "mvar": "v333"
             }
            ],
            "attrs": {},
            "op": "mul"
           }
          ],
          "attrs": {},
          "op": "add"
         }
        ],
        "attrs": {},
        "op": "mul"
       },
       {
        "args": [
         {
          "mvar": "v334"
         },
         {
          "mvar": "v335"
         }
        ],
        "attrs": {},
        "op": "mul"
       }
      ],
      "attrs": {},
      "op": "add"
     }
    ],
    "attrs": {},
    "op": "mul"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "fold": {
   "bound_norm": "spectral",
   "cert": null,
   "cond": null,
   "derivation": [],
   "dspec": null,
   "error_bound": null,
   "kind": "law",
   "law": "",
   "lhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "args": [
       {
        "mvar": "v323"
       },
       {
        "mvar": "v324"
       },
       {
        "mvar": "v325"
       },
       {
        "mvar": "v326"
       },
       {
        "mvar": "v327"
       },
       {
        "mvar": "v328"
       },
       {
        "mvar": "v329"
       },
       {
        "mvar": "v330"
       },
       {
        "mvar": "v331"
       },
       {
        "mvar": "v332"
       },
       {
        "mvar": "v333"
       },
       {
        "mvar": "v334"
       },
       {
        "mvar": "v335"
       }
      ],
      "attrs": {},
      "op": "synop2_35bbd03301"
     }
    ],
    "attrs": {},
    "op": "mul"
   },
   "missing_hooks": [],
   "name": "prim:synop3_fbbba382fc",
   "rhs": {
    "args": [
     {
      "mvar": "v10"
     },
     {
      "mvar": "v323"
     },
     {
      "mvar": "v324"
     },
     {
      "mvar": "v325"
     },
     {
      "mvar": "v326"
     },
     {
      "mvar": "v327"
     },
     {
      "mvar": "v328"
     },
     {
      "mvar": "v329"
     },
     {
      "mvar": "v330"
     },
     {
      "mvar": "v331"
     },
     {
      "mvar": "v332"
     },
     {
      "mvar": "v333"
     },
     {
      "mvar": "v334"
     },
     {
      "mvar": "v335"
     }
    ],
    "attrs": {},
    "op": "synop3_fbbba382fc"
   },
   "serializable": true,
   "tags": [],
   "version": 2
  },
  "handler": {
   "family": "ew",
   "max_err": 0.0,
   "record": {
    "args": [
     "v10",
     "v323",
     "v324",
     "v325",
     "v326",
     "v327",
     "v328",
     "v329",
     "v330",
     "v331",
     "v332",
     "v333",
     "v334",
     "v335"
    ],
    "handler_body": [
     "ew",
     [
      "mul",
      "v10",
      [
       "add",
       [
        "mul",
        "v323",
        [
         "add",
         [
          "mul",
          "v324",
          [
           "add",
           [
            "mul",
            "v325",
            [
             "add",
             [
              "mul",
              "v326",
              "v327"
             ],
             [
              "mul",
              "v328",
              "v329"
             ]
            ]
           ],
           [
            "mul",
            "v330",
            "v331"
           ]
          ]
         ],
         [
          "mul",
          "v332",
          "v333"
         ]
        ]
       ],
       [
        "mul",
        "v334",
        "v335"
       ]
      ]
     ]
    ],
    "kernel": "minted_0_k",
    "kind": "handler",
    "pattern": [
     "mul",
     "v10",
     [
      "add",
      [
       "mul",
       "v323",
       [
        "add",
        [
         "mul",
         "v324",
         [
          "add",
          [
           "mul",
           "v325",
           [
            "add",
            [
             "mul",
             "v326",
             "v327"
            ],
            [
             "mul",
             "v328",
             "v329"
            ]
           ]
          ],
          [
           "mul",
           "v330",
           "v331"
          ]
         ]
        ],
        [
         "mul",
         "v332",
         "v333"
        ]
       ]
      ],
      [
       "mul",
       "v334",
       "v335"
      ]
     ]
    ],
    "provenance": {
     "atol": 1e-12,
     "budget": 2048,
     "domain": [
      {
       "v10": [
        2,
        16,
        4
       ],
       "v323": [
        2,
        16,
        4
       ],
       "v324": [
        2,
        16,
        4
       ],
       "v325": [
        2,
        16,
        4
       ],
       "v326": [
        2,
        16,
        4
       ],
       "v327": [
        2,
        16,
        4
       ],
       "v328": [
        2,
        16,
        1
       ],
       "v329": [
        2,
        1,
        4
       ],
       "v330": [
        2,
        16,
        1
       ],
       "v331": [
        2,
        1,
        4
       ],
       "v332": [
        2,
        16,
        1
       ],
       "v333": [
        2,
        1,
        4
       ],
       "v334": [
        2,
        16,
        1
       ],
       "v335": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v323": [
        2,
        16,
        4
       ],
       "v324": [
        2,
        16,
        4
       ],
       "v325": [
        2,
        16,
        4
       ],
       "v326": [
        2,
        16,
        4
       ],
       "v327": [
        2,
        16,
        4
       ],
       "v328": [
        2,
        16,
        1
       ],
       "v329": [
        2,
        1,
        4
       ],
       "v330": [
        2,
        16,
        1
       ],
       "v331": [
        2,
        1,
        4
       ],
       "v332": [
        2,
        16,
        1
       ],
       "v333": [
        2,
        1,
        4
       ],
       "v334": [
        2,
        16,
        1
       ],
       "v335": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v323": [
        2,
        16,
        4
       ],
       "v324": [
        2,
        16,
        4
       ],
       "v325": [
        2,
        16,
        4
       ],
       "v326": [
        2,
        16,
        4
       ],
       "v327": [
        2,
        16,
        4
       ],
       "v328": [
        2,
        16,
        1
       ],
       "v329": [
        2,
        1,
        4
       ],
       "v330": [
        2,
        16,
        1
       ],
       "v331": [
        2,
        1,
        4
       ],
       "v332": [
        2,
        16,
        1
       ],
       "v333": [
        2,
        1,
        4
       ],
       "v334": [
        2,
        16,
        1
       ],
       "v335": [
        2,
        1,
        4
       ]
      },
      {
       "v10": [
        2,
        16,
        4
       ],
       "v323": [
        2,
        16,
        4
       ],
       "v324": [
        2,
        16,
        4
       ],
       "v325": [
        2,
        16,
        4
       ],
       "v326": [
        2,
        16,
        4
       ],
       "v327": [
        2,
        16,
        4
       ],
       "v328": [
        2,
        16,
        1
       ],
       "v329": [
        2,
        1,
        4
       ],
       "v330": [
        2,
        16,
        1
       ],
       "v331": [
        2,
        1,
        4
       ],
       "v332": [
        2,
        16,
        1
       ],
       "v333": [
        2,
        1,
        4
       ],
       "v334": [
        2,
        16,
        1
       ],
       "v335": [
        2,
        1,
        4
       ]
      }
     ],
     "family": "ew",
     "grammar": "handler-grammar-v4",
     "max_err": 0.0,
     "n_samples": 8,
     "origin": "minted",
     "rtol": 1e-09,
     "seed": 0,
     "tried": 1
    },
    "tag": "minted_0"
   },
   "tried": 1,
   "verdict": "minted"
  },
  "mods": [
   "GatedLinearAttention",
   "MambaSelectiveScan",
   "RealGatedLRU"
  ],
  "sites": 12
 }
}"""
)

#: The minted op names the prim family ships — all three depths:
#: ``synop_<sha256[:10]>`` (the zoo-prim mint), the depth-2
#: ``synop2_<sha256[:10]>`` (the vocabulary-bootstrap mint) and the
#: depth-3 ``synop3_<sha256[:10]>`` (the vocab-priced bootstrap).
#: A ``synop_*`` op name means "a shipped machine-minted primitive"
#: wherever this set is consulted — expansion, denotation lookup,
#: provenance.
SYNOP_SHIPPED: frozenset[str] = (
    frozenset(_PRIM_RECORDS)
    | frozenset(_PRIM2_RECORDS)
    | frozenset(_PRIM3_RECORDS)
)

#: The depth-2 subset — the bootstrap mint's ``synop2_*`` ops.
SYNOP2_SHIPPED: frozenset[str] = frozenset(_PRIM2_RECORDS)

#: The depth-3 subset — the vocab-priced bootstrap's ``synop3_*``
#: ops (skeletons spelled through both shipped tiers).
SYNOP3_SHIPPED: frozenset[str] = frozenset(_PRIM3_RECORDS)

#: Law text for the fold rules — a structural fold collapsing a
#: consumer pattern into one kernel, same class as ``conv_bn_fold``.
_PRIM_FOLD_LAW = (
    "machine-minted primitive fold — the manual subgraph collapses "
    "to the {synop} primitive (zoo-prim mint; admitted by the full "
    "gauntlet, machine-original; denotation definitional — the op's "
    "torch binding IS the spelled subgraph)"
)

#: Law text for the definitional premise — the cert leg's anchor.
_PRIM_EXPAND_LAW = (
    "{synop} unfolds to its mined denotation — the definitional "
    "premise the cert leg's anchored derivation unions in when the "
    "bare bank cannot derive the fold"
)


def synop_name(fold_name: str) -> str:
    """Return the op a ``prim:synop_*`` fold name mints (identity otherwise)."""
    return fold_name[5:] if fold_name.startswith("prim:") else fold_name


def expand_name(synop: str) -> str:
    """Return the definitional-premise law name for a ``synop_*`` op."""
    return f"{synop_name(synop)}.expand"


def prim_records() -> dict[str, Any]:
    """Return the embedded prim records — the frozen mint artifacts.

    All three depths, merged: the 21 depth-1 ``synop_*`` records plus
    the 2 depth-2 ``synop2_*`` records plus the 34 depth-3
    ``synop3_*`` records (the deeper tiers additionally carry
    ``alpha_fold`` / ``embeds_shipped`` / ``embeds`` / ``expand``).
    """
    return {**_PRIM_RECORDS, **_PRIM2_RECORDS, **_PRIM3_RECORDS}


def prim2_records() -> dict[str, Any]:
    """Return the embedded depth-2 records — the bootstrap mint's."""
    return _PRIM2_RECORDS


def prim3_records() -> dict[str, Any]:
    """Return the embedded depth-3 records — the vocab-priced mint's."""
    return _PRIM3_RECORDS


_PRIM_MATERIALIZED: dict[str, tuple[Rewrite, Rewrite]] = {}


def prim_fold(synop: str) -> tuple[Rewrite, Rewrite]:
    """Build ``(fold, expand)`` for one prim record — memoized.

    The fold rebuilds through the shipped record codec
    (:func:`~mlopt_core.laws.serialize.object_from_data`) — the same
    object the mint admitted — then gets the shipped integration
    fields (``law`` text, ``FUSION`` tag) and its premise attached,
    exactly as ``pipeline.Proposal.as_rule`` minted it; the op's
    :func:`~mlopt_core.ir.op_def` declaration registers with the fold
    (the generator entry is torch-free — the *denotation* is the
    domain-side piece).  Interned: the same objects come back on
    every call.

    *synop* resolves across all three tables — a ``synopN_*`` name
    materializes its depth-N record.  The deeper-tier records embed
    the definitional premise verbatim (``rec["expand"]``): its RHS is
    the skeleton *re-expanded to manual ops*, NOT the fold's LHS —
    the cert leg's anchored derivation then re-folds through the
    shipped lower-tier ``prim:synop*`` folds (an N-step anchored
    derivation; the lower-tier folds are bank members, not further
    premises).  A record whose embedded premise does not start at the
    fold's own call raises rather than materializing a mismatched
    axiom.
    """
    if synop in _PRIM_MATERIALIZED:
        return _PRIM_MATERIALIZED[synop]
    if synop in _PRIM_RECORDS:
        rec = _PRIM_RECORDS[synop]
    elif synop in _PRIM2_RECORDS:
        rec = _PRIM2_RECORDS[synop]
    elif synop in _PRIM3_RECORDS:
        rec = _PRIM3_RECORDS[synop]
    else:
        raise KeyError(synop)
    fold = object_from_data(rec["fold"])
    if "expand" in rec:
        # depth-2+: the premise's RHS is the full manual denotation —
        # verbatim from the mint, not rebuildable as ``rhs -> lhs``.
        ex = object_from_data(rec["expand"])
        if ex.lhs != fold.rhs:
            raise ValueError(
                f"prim record {synop!r} embeds an expand premise "
                f"whose LHS is not the fold's own call — a "
                f"mismatched axiom would anchor nothing"
            )
        denotation = ex.rhs
    else:
        denotation = fold.lhs
    expand = Rewrite(
        expand_name(synop),
        fold.rhs,
        denotation,
        law=_PRIM_EXPAND_LAW.format(synop=synop),
        tags=frozenset({tags.SIMPLIFICATION}),
    )
    out = Rewrite(
        name=fold.name,
        lhs=fold.lhs,
        rhs=fold.rhs,
        law=_PRIM_FOLD_LAW.format(synop=synop),
        tags=frozenset({tags.FUSION}),
        error_bound=fold.error_bound,
        bound_norm=fold.bound_norm,
        derivation=fold.derivation,
        cond=fold.cond,
        dspec=fold.dspec,
    )
    object.__setattr__(out, "premises", (expand,))
    _PRIM_MATERIALIZED[synop] = (out, expand)
    op_def(
        synop,
        len(fold.rhs.args),
        1,
        law=(
            f"synthesized primitive — denotation is the mined manual "
            f"subgraph {rec['alpha']}"
        ),
    )
    return _PRIM_MATERIALIZED[synop]


def prim_folds() -> tuple[Rewrite, ...]:
    """Return the shipped fold laws — ``pattern -> synop*_*(args)``.

    All three depths: the 21 ``prim:synop_*`` folds, then the 2
    depth-2 ``prim:synop2_*`` folds, then the 34 depth-3
    ``prim:synop3_*`` folds (skeleton -> ``synopN_*(args)``).
    """
    return tuple(
        prim_fold(n)[0]
        for n in (*_PRIM_RECORDS, *_PRIM2_RECORDS, *_PRIM3_RECORDS)
    )


def prim2_folds() -> tuple[Rewrite, ...]:
    """Return the 2 depth-2 fold laws — the bootstrap mint's slice."""
    return tuple(prim_fold(n)[0] for n in _PRIM2_RECORDS)


def prim3_folds() -> tuple[Rewrite, ...]:
    """Return the 34 depth-3 fold laws — the vocab-priced mint's."""
    return tuple(prim_fold(n)[0] for n in _PRIM3_RECORDS)


def prim_premises() -> tuple[Rewrite, ...]:
    """Return the definitional ``synop*_*.expand`` premises — axioms.

    NOT a saturation set: expanding ``synop -> pattern`` inside a
    search would re-add the unfolded spelling the fold collapses.
    These are the anchored-derivation axioms a cert/verify leg unions
    into its axiom set; each fold also carries its own premise as
    ``rule.premises``.  All three depths — a deeper-tier premise's
    RHS is the skeleton's *expanded* manual denotation (the embedded
    ``expand`` record), so its anchored derivation still closes
    through the shipped lower-tier folds.
    """
    return tuple(
        prim_fold(n)[1]
        for n in (*_PRIM_RECORDS, *_PRIM2_RECORDS, *_PRIM3_RECORDS)
    )


def prim2_premises() -> tuple[Rewrite, ...]:
    """Return the depth-2 ``synop2_*.expand`` premises — cert axioms."""
    return tuple(prim_fold(n)[1] for n in _PRIM2_RECORDS)


def prim3_premises() -> tuple[Rewrite, ...]:
    """Return the depth-3 ``synop3_*.expand`` premises — cert axioms."""
    return tuple(prim_fold(n)[1] for n in _PRIM3_RECORDS)


def _resolve(name: str, lib: dict[str, Rewrite], seen: frozenset[str]):
    """Rebuild *name*'s Rewrite; return ``(rule, transported)``.

    *transported* names the hooks the ``"guard"`` spec restored —
    ``()`` for a library-resolved leaf (a live object is its own
    full-strength declaration) or an unguarded record.  Raises
    ``ValueError`` on any shortfall the store path would degrade
    through: shipping a weakened composite is a smuggle, not a
    partial admit.
    """
    rec = _RECORDS.get(name)
    if rec is None:
        try:
            return lib[name], ()
        except KeyError:
            raise ValueError(
                f"minted object {name!r} names a parent that is "
                f"neither an embedded record nor a library rule"
            ) from None
    if name in seen:
        raise ValueError(f"cyclic guard spec at {name!r}")
    rule = object_from_data(rec)
    guard = rec.get("guard")
    if not isinstance(guard, dict):
        return rule, ()
    parents = tuple(guard.get("parents") or ())
    if len(parents) != 2:
        raise ValueError(
            f"minted object {name!r} carries a guard spec with "
            f"{len(parents)} parents — composition is binary"
        )
    try:
        pats = (
            _dec_binding(guard["pat1"]),
            _dec_binding(guard["pat2"]),
        )
    except Exception as exc:
        raise ValueError(
            f"minted object {name!r} carries an undecodable guard spec"
        ) from exc
    resolved = [
        _resolve_parent(pname, lib, seen | {name}) for pname in parents
    ]
    return _compose_record(name, rule, resolved, parents, pats)


def _resolve_parent(
    pname: str, lib: dict[str, Rewrite], seen: frozenset[str]
) -> Rewrite:
    """Resolve one guard parent — record first (recursive), then lib.

    A record-resolved parent must rebuild at *full declared strength*
    (``missing_hooks`` minus transported empty) — the same bar
    ``evidence._rebuild_stored`` holds, raised here into an error
    rather than a silent degrade.
    """
    prule, ptrans = _resolve(pname, lib, seen)
    prec = _RECORDS.get(pname)
    if prec is not None and set(prec.get("missing_hooks") or ()) - set(
        ptrans
    ):
        raise ValueError(
            f"minted parent {pname!r} rebuilt weaker than it "
            f"declared (missing "
            f"{sorted(set(prec['missing_hooks']) - set(ptrans))}) "
            f"— composing over it would smuggle a weakened guard"
        )
    return prule


def _compose_record(
    name: str,
    rule: Rewrite,
    resolved: list[Rewrite],
    parents: tuple[str, ...],
    pats: tuple[Any, Any],
):
    """Re-run ``_compose_guards`` over resolved parents; rewrap *rule*.

    Returns ``(out, transported)`` — *transported* names the hooks the
    guard spec restored.  Raises ``ValueError`` when the spec yields no
    hooks (a guard-less rebuild is not the declared object).
    """
    check, drv = meta._compose_guards(
        resolved[0], resolved[1], pats[0], pats[1]
    )
    if check is None and drv is None:
        raise ValueError(
            f"minted object {name!r} declares a guard spec but "
            f"transports no hooks"
        )
    # Same policy as synthesize_rules.offer / rulecache._dec_rule: the
    # composite derive attaches only where the RHS carries the "@i:"
    # placeholders it refills.
    if not (
        drv is not None and meta._rhs_derive_placeholders(rule.rhs)
    ):
        drv = None
    out = Rewrite(
        name=rule.name,
        lhs=rule.lhs,
        rhs=rule.rhs,
        law=rule.law,
        tags=rule.tags,
        error_bound=rule.error_bound,
        bound_norm=rule.bound_norm,
        derivation=rule.derivation,
        cond=rule.cond,
        dspec=rule.dspec,
        check=check,
        derive=drv,
    )
    object.__setattr__(out, "parents", parents)
    object.__setattr__(out, "guard_pats", pats)
    meta.SYNTH_PARENTS[out.name] = (parents[0], parents[1])
    transported = tuple(
        h
        for h, f in (("check", check), ("derive", drv))
        if f is not None
    )
    return out, transported


def materialize(
    name: str,
    lib: dict[str, Rewrite],
    *,
    ship_as: str | None = None,
    tags: Any = None,
    law: str | None = None,
) -> Rewrite:
    """Materialize an embedded minted record as a shipped ``Rewrite``.

    *name* is the minted record's own (``syn_*``) name; *lib* is the
    name→rule table the leaf parents resolve against (the shipped
    library families — e.g. ``BATCH_NORM_RULES`` + ``CONV_FOLD_RULES``).
    *ship_as* renames the root object for the library (promotion
    practice: minted ``syn_*`` names ship under their law name, as the
    promoted-laws table did); *tags* / *law* override the record's
    fields for the shipped object — integration decisions, recorded
    here rather than smuggled into the verbatim record.  The
    intermediate nodes keep their minted names: they are scaffolding,
    not shipped rules.

    The returned object carries ``parents`` / ``guard_pats`` — the
    same provenance ``meta.synthesize_rules`` mints carry — so the
    store's ``_guard_transport_spec`` re-encodes it faithfully on a
    re-store, and ``missing_hooks`` on it honestly reports the
    procedural hooks its class carries.
    """
    rule, _ = _resolve(name, lib, frozenset())
    if ship_as is not None:
        parents = getattr(rule, "parents", None)
        object.__setattr__(rule, "name", ship_as)
        if parents is not None:
            meta.SYNTH_PARENTS[ship_as] = parents
    if tags is not None:
        object.__setattr__(rule, "tags", frozenset(tags))
    if law is not None:
        object.__setattr__(rule, "law", law)
    return rule
