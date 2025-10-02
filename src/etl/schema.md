# Graph Schema (CKG + PubMedKG)

## Node labels

| Label       | `~id` example      | Key properties                         |
|-------------|--------------------|----------------------------------------|
| Gene        | `EGFR`             | `hgnc_id` (optional)                   |
| Protein     | `EGFR`             | none                                   |
| Disease     | `Colon Cancer`     | none                                   |
| Experiment  | `EXP001`           | `name`, `technology`, `disease`        |
| Sample      | `S001`             | `disease`, `experiment_id`             |
| Measurement | (edge-only)        | values stored as edge properties       |
| Finding     | `FIND001`          | `finding_type`, `description`          |
| Pathway     | `PW001`            | `name`, `genes`                        |
| Publication | `12345678` (PMID)  | `title`, `journal`, `year`             |

## Edge labels

| Edge label        | `~from` → `~to`                | Properties                                  |
|-------------------|-------------------------------|---------------------------------------------|
| `ENCODES`         | Gene → Protein                | none                                        |
| `MEASURES`        | Experiment → Protein          | `measurement_type`, `value`, `unit`, `p_value` |
| `COLLECTED_FROM`  | Experiment → Sample           | none                                        |
| `FROM_DISEASE`    | Sample → Disease              | none                                        |
| `DERIVED_FROM`    | Finding → Experiment          | none                                        |
| `IMPLICATES`      | Finding → Entity              | `entity_type`                               |
| `INVOLVES`        | Pathway → Gene/Protein/Finding| none                                        |
| `INTERACTS_WITH`  | Protein → Protein             | `evidence_source`                           |
| `MENTIONS`        | Publication → Entity          | `entity_type`, `snippet`                    |
| `CITES`           | Publication → Publication     | none                                        |

## Validation queries (openCypher)

```cypher
// Count nodes per label
MATCH (n) RETURN labels(n) AS label, count(*) AS count;

// Proteins measured in a specific experiment
MATCH (e:Experiment {`~id`: 'EXP001'})-[:MEASURES]->(p:Protein)
RETURN p.`~id`, e.name;

// Findings linked to diseases
MATCH (f:Finding)-[:IMPLICATES]->(d:Disease)
RETURN f.`~id`, f.description, d.`~id`;

// Publications mentioning EGFR
MATCH (pub:Publication)-[m:MENTIONS]->(entity {`~id`: 'EGFR'})
RETURN pub.`~id`, pub.title, m.snippet;
```
# Graph Schema Specification

This document captures the node/edge definitions used for Amazon Neptune bulk
loading (openCypher CSV format) for the GraphRAG biomedical project.

## Node labels and properties

| Label        | `~id` format                      | Key properties                         |
|--------------|-----------------------------------|----------------------------------------|
| `Gene`       | HGNC symbol (e.g., `BRAF`)        | `hgnc_id`                              |
| `Variant`    | `VAR_<index>`                     | `gene_symbol`, `sample_id`, `variant_class`, `protein_change`, `Project`, `civic_variant_id` |
| `Sample`     | TCGA barcode (e.g., `TCGA-AB-...`) | `project_id`, `disease`, `source`      |
| `Evidence`   | CIVIC evidence ID                 | `source_type`, `evidence_level`, `evidence_type`, `description`, `clinical_significance`, `rating`, `citation_id`, `citation_text` |
| `Publication`| PubMed PMID                        | `title`, `journal`, `year`             |
| `Disease`    | Disease label (string)            | none                                   |
| `Drug`       | Drug label (string)               | none                                   |

## Edge labels and properties

| Edge label           | `~from`                | `~to`                      | Properties                       |
|----------------------|------------------------|---------------------------|----------------------------------|
| `HAS_VARIANT`        | `Gene`                 | `Variant`                 | none                             |
| `HAS_VARIANT_SAMPLE` | `Sample`               | `Variant`                 | none                             |
| `FROM_DISEASE`       | `Sample`               | `Disease`                 | none                             |
| `ASSOCIATED_WITH`    | `Variant`              | `Disease`                 | none                             |
| `SUPPORTS_VARIANT`   | `Evidence`             | `Variant`                 | none                             |
| `SUPPORTS_DISEASE`   | `Evidence`             | `Disease`                 | none                             |
| `MENTIONS`           | `Publication`          | Entity ID (Gene/Drug/etc) | `entity_type`, `evidence_text`   |
| `CITES`              | `Publication`          | `Publication`             | none                             |

## Sample validation queries (openCypher)

```cypher
// Count nodes by label
MATCH (n) RETURN labels(n) AS label, count(*) AS count;

// List variants associated with melanoma
MATCH (v:Variant)-[:ASSOCIATED_WITH]->(d:Disease {`~id`: "Melanoma"})
RETURN v.`~id`, v.gene_symbol, v.protein_change;

// Evidence supporting a variant-disease link
MATCH (e:Evidence)-[:SUPPORTS_VARIANT]->(v:Variant {`~id`: "VAR_0"})
OPTIONAL MATCH (e)-[:SUPPORTS_DISEASE]->(d:Disease)
RETURN e.`~id`, e.description, d.`~id`;

// Publications mentioning EGFR
MATCH (p:Publication)-[:MENTIONS]->(entity {`~id`: "EGFR"})
RETURN p.`~id`, p.title, entity.`~id`, labels(entity);
```




