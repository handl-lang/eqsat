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
    "SYNOP_SHIPPED",
    "expand_name",
    "materialize",
    "prim2_folds",
    "prim2_premises",
    "prim2_records",
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

#: The minted op names the prim family ships — both depths:
#: ``synop_<sha256[:10]>`` (the zoo-prim mint) and the depth-2
#: ``synop2_<sha256[:10]>`` (the vocabulary-bootstrap mint).  A
#: ``synop_*`` op name means "a shipped machine-minted primitive"
#: wherever this set is consulted — expansion, denotation lookup,
#: provenance.
SYNOP_SHIPPED: frozenset[str] = frozenset(_PRIM_RECORDS) | frozenset(
    _PRIM2_RECORDS
)

#: The depth-2 subset — the bootstrap mint's ``synop2_*`` ops.
SYNOP2_SHIPPED: frozenset[str] = frozenset(_PRIM2_RECORDS)

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

    Both depths, merged: the 21 depth-1 ``synop_*`` records plus the
    2 depth-2 ``synop2_*`` records (the latter additionally carry
    ``alpha_fold`` / ``embeds_shipped`` / ``embeds`` / ``expand``).
    """
    return {**_PRIM_RECORDS, **_PRIM2_RECORDS}


def prim2_records() -> dict[str, Any]:
    """Return the embedded depth-2 records — the bootstrap mint's."""
    return _PRIM2_RECORDS


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

    *synop* resolves across both tables — a ``synop2_*`` name
    materializes its depth-2 record.  The depth-2 record embeds the
    definitional premise verbatim (``rec["expand"]``): its RHS is the
    skeleton *re-expanded to manual ops*, NOT the fold's LHS — the
    cert leg's anchored derivation then re-folds through the shipped
    depth-1 ``prim:synop_*`` (a 2-step anchored derivation; the
    depth-1 fold is a bank member, not a second premise).  A record
    whose embedded premise does not start at the fold's own call
    raises rather than materializing a mismatched axiom.
    """
    if synop in _PRIM_MATERIALIZED:
        return _PRIM_MATERIALIZED[synop]
    if synop in _PRIM_RECORDS:
        rec = _PRIM_RECORDS[synop]
    elif synop in _PRIM2_RECORDS:
        rec = _PRIM2_RECORDS[synop]
    else:
        raise KeyError(synop)
    fold = object_from_data(rec["fold"])
    if "expand" in rec:
        # depth-2: the premise's RHS is the full manual denotation —
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
    """Return the shipped fold laws — ``pattern -> synop_*(args)``.

    Both depths: the 21 ``prim:synop_*`` folds then the 2 depth-2
    ``prim:synop2_*`` folds (skeleton -> ``synop2_*(args)``).
    """
    return tuple(
        prim_fold(n)[0] for n in (*_PRIM_RECORDS, *_PRIM2_RECORDS)
    )


def prim2_folds() -> tuple[Rewrite, ...]:
    """Return the 2 depth-2 fold laws — the bootstrap mint's slice."""
    return tuple(prim_fold(n)[0] for n in _PRIM2_RECORDS)


def prim_premises() -> tuple[Rewrite, ...]:
    """Return the definitional ``synop_*.expand`` premises — cert axioms.

    NOT a saturation set: expanding ``synop -> pattern`` inside a
    search would re-add the unfolded spelling the fold collapses.
    These are the anchored-derivation axioms a cert/verify leg unions
    into its axiom set; each fold also carries its own premise as
    ``rule.premises``.  Both depths — a depth-2 premise's RHS is the
    skeleton's *expanded* manual denotation (the embedded
    ``expand`` record), so its anchored derivation still closes
    through the shipped depth-1 fold.
    """
    return tuple(
        prim_fold(n)[1] for n in (*_PRIM_RECORDS, *_PRIM2_RECORDS)
    )


def prim2_premises() -> tuple[Rewrite, ...]:
    """Return the depth-2 ``synop2_*.expand`` premises — cert axioms."""
    return tuple(prim_fold(n)[1] for n in _PRIM2_RECORDS)


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
