"""Generate the sample VTU notes PDFs used by the demo, tests and eval set.

    python scripts/make_sample_pdfs.py

Writes data/pdfs/DBMS/DBMS_Module2.pdf and data/pdfs/OS/OS_Module3.pdf.
Page numbers here are what eval/questions.json expects, so keep them in sync.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf  # type: ignore[no-redef]

from src.config import load_settings  # noqa: E402

PAGE_W, PAGE_H = 595, 842  # A4 points
MARGIN = 56

DBMS_PAGES: list[tuple[str, str]] = [
    (
        "DBMS - Module 2: Relational Database Design",
        """Visvesvaraya Technological University
Semester 5 - Database Management Systems (BCS403)
Module 2 notes - Relational Database Design and Normalization

Contents
Page 2 - Functional dependency and its types
Page 3 - Armstrong axioms and attribute closure
Page 4 - Normalization, First Normal Form, Second Normal Form
Page 5 - Third Normal Form and Boyce-Codd Normal Form
Page 6 - Lossless join decomposition and dependency preservation
Page 7 - Multivalued dependency and Fourth Normal Form
Page 8 - Denormalization and module summary

These notes follow the prescribed VTU syllabus and are meant for revision
before the semester end examination.""",
    ),
    (
        "2.1 Functional Dependency",
        """A functional dependency is a constraint between two sets of attributes in a
relation. For a relation R, the functional dependency X -> Y holds if and only if
for every pair of tuples t1 and t2 in R, whenever t1[X] = t2[X] it is also true
that t1[Y] = t2[Y]. X is called the determinant and Y is called the dependent.

Types of functional dependency

1. Trivial functional dependency: X -> Y is trivial when Y is a subset of X.
   For example, {StudentID, Name} -> Name is trivial.

2. Non-trivial functional dependency: X -> Y where Y is not a subset of X.
   For example, StudentID -> Name.

3. Full functional dependency: Y is fully functionally dependent on X if Y is
   functionally dependent on X but not on any proper subset of X.

4. Partial functional dependency: a non-prime attribute depends on part of a
   composite candidate key. Partial dependency is what Second Normal Form
   removes.

5. Transitive functional dependency: if X -> Y and Y -> Z hold, and Y is not a
   candidate key, then X -> Z is a transitive dependency. Third Normal Form
   removes transitive dependency of non-prime attributes on the key.

A prime attribute is an attribute that is part of some candidate key. Every
other attribute is a non-prime attribute. Functional dependencies are the basis
of all normalization work, because they tell us which decomposition preserves
the meaning of the data.""",
    ),
    (
        "2.2 Armstrong Axioms and Attribute Closure",
        """Armstrong axioms are a sound and complete set of inference rules used to
derive all functional dependencies logically implied by a given set F.

The three primary axioms are:

1. Reflexivity: if Y is a subset of X, then X -> Y.
2. Augmentation: if X -> Y holds, then XZ -> YZ holds for any set Z.
3. Transitivity: if X -> Y and Y -> Z hold, then X -> Z holds.

Three secondary rules are derived from the primary axioms:

4. Union: if X -> Y and X -> Z, then X -> YZ.
5. Decomposition: if X -> YZ, then X -> Y and X -> Z.
6. Pseudo-transitivity: if X -> Y and WY -> Z, then WX -> Z.

The set of all dependencies derivable from F is called the closure of F,
written F+. Armstrong axioms are sound because they generate only dependencies
that F implies, and complete because they generate every such dependency.

Attribute closure algorithm

To compute X+, the closure of attribute set X under F: start with result = X,
then repeatedly, for every dependency A -> B in F with A a subset of result,
add B to result, until result stops changing. X is a superkey of R if and only
if X+ contains every attribute of R. This test is the standard way to find
candidate keys in the examination.""",
    ),
    (
        "2.3 Normalization, 1NF and 2NF",
        """Normalization is the process of decomposing a relation into smaller relations
so that redundancy and update, insert and delete anomalies are reduced, while
the information content of the original relation is preserved.

First Normal Form (1NF)
A relation is in First Normal Form if every attribute holds only atomic
(indivisible) values and there are no repeating groups or multi-valued
attributes. A table storing several phone numbers in one column violates 1NF;
the fix is to place each phone number in its own tuple or in a separate
relation.

Second Normal Form (2NF)
A relation is in Second Normal Form if it is already in 1NF and every non-prime
attribute is fully functionally dependent on the whole of every candidate key.
In other words, 2NF removes partial dependency on a composite key.

Example: consider SCORE(StudentID, CourseID, Marks, StudentName) with the
candidate key {StudentID, CourseID}. The dependency StudentID -> StudentName is
a partial dependency, because StudentName depends on only part of the key. The
relation is therefore not in 2NF. Decompose it into
STUDENT(StudentID, StudentName) and SCORE(StudentID, CourseID, Marks).

Note that a relation whose candidate keys are all single attributes is
automatically in 2NF, since partial dependency is impossible.""",
    ),
    (
        "2.4 Third Normal Form and BCNF",
        """Third Normal Form (3NF)
A relation R is in Third Normal Form if it is in 2NF and no non-prime attribute
is transitively dependent on any candidate key. Equivalently, for every
non-trivial dependency X -> Y in R, either X is a superkey of R, or every
attribute of Y is a prime attribute.

Example: EMPLOYEE(EmpID, DeptID, DeptName) with EmpID -> DeptID and
DeptID -> DeptName. DeptName is transitively dependent on EmpID, so the
relation is not in 3NF. Split it into EMPLOYEE(EmpID, DeptID) and
DEPARTMENT(DeptID, DeptName).

Boyce-Codd Normal Form (BCNF)
A relation is in BCNF if for every non-trivial functional dependency X -> Y,
the determinant X is a superkey. BCNF is a stricter version of 3NF: it removes
the exception that allows Y to consist of prime attributes.

Difference between 3NF and BCNF
1. 3NF allows a non-trivial dependency X -> Y where X is not a superkey,
   provided Y is a prime attribute. BCNF does not allow this at all.
2. Every relation in BCNF is in 3NF, but a relation in 3NF need not be in BCNF.
3. A 3NF decomposition that is both lossless and dependency preserving always
   exists. A BCNF decomposition is always lossless, but it may fail to preserve
   all functional dependencies.
4. BCNF removes redundancy caused by overlapping candidate keys, which 3NF can
   leave behind.""",
    ),
    (
        "2.5 Lossless Join Decomposition and Dependency Preservation",
        """When a relation R is decomposed into R1 and R2, two properties decide whether
the decomposition is acceptable.

Lossless join decomposition
A decomposition of R into R1 and R2 is a lossless join decomposition if the
natural join of R1 and R2 gives exactly the original relation R, with no
spurious tuples. The binary test is: the decomposition is lossless if and only
if the common attributes of R1 and R2 form a superkey of at least one of them,
that is (R1 intersect R2) -> R1 or (R1 intersect R2) -> R2 holds in F+.

A decomposition that is not lossless is called a lossy join decomposition,
because the natural join produces extra tuples and information is lost.

Dependency preservation
A decomposition is dependency preserving if the union of the functional
dependencies that can be checked on the individual sub-relations logically
implies every dependency in F, that is (F1 union F2 union ... ) + = F+.
Dependency preservation matters because it lets the database enforce every
constraint without computing a join.

Summary of guarantees
- Decomposition into 3NF: lossless and dependency preserving.
- Decomposition into BCNF: lossless, but dependency preservation is not
  guaranteed.""",
    ),
    (
        "2.6 Multivalued Dependency and 4NF",
        """A multivalued dependency X ->> Y holds in relation R when, for each value of
X, the set of values of Y is independent of the remaining attributes of R. A
multivalued dependency represents a situation where two independent
many-to-many facts are stored in the same relation, which causes a large amount
of redundancy.

Example: STUDENT(StudentID, Course, Hobby) where a student takes several
courses and has several hobbies that are unrelated to those courses. Here
StudentID ->> Course and StudentID ->> Hobby. Each student then needs one tuple
for every combination of course and hobby.

Fourth Normal Form (4NF)
A relation is in Fourth Normal Form if it is in BCNF and, for every non-trivial
multivalued dependency X ->> Y, X is a superkey of the relation. The fix for
the example above is to decompose it into STUDENT_COURSE(StudentID, Course) and
STUDENT_HOBBY(StudentID, Hobby), removing the cross product of unrelated facts.

Join dependency and 5NF
A join dependency generalises the idea to decomposition into three or more
relations. A relation is in Fifth Normal Form, also called Project Join Normal
Form, when every join dependency in it is implied by its candidate keys.""",
    ),
    (
        "2.7 Denormalization and Module Summary",
        """Denormalization is the deliberate reintroduction of redundancy into a
normalized schema to improve read performance. It trades write cost and the
risk of anomalies for fewer joins at query time, and is used mainly in
reporting and data warehouse workloads.

Module summary
1NF - atomic values, no repeating groups.
2NF - 1NF plus no partial dependency on a composite candidate key.
3NF - 2NF plus no transitive dependency of a non-prime attribute.
BCNF - every determinant is a superkey.
4NF - BCNF plus every non-trivial multivalued dependency has a superkey
determinant.
5NF - every join dependency is implied by the candidate keys.

Typical examination questions
1. Define functional dependency and explain its types. (5 marks)
2. State and prove Armstrong axioms. (10 marks)
3. Explain 3NF and BCNF with an example, and bring out the difference.
   (10 marks)
4. Define lossless join decomposition and give the test for it. (5 marks)""",
    ),
]

OS_PAGES: list[tuple[str, str]] = [
    (
        "Operating Systems - Module 3",
        """Visvesvaraya Technological University
Semester 4 - Operating Systems (BCS303)
Module 3 notes - Deadlocks and Memory Management

Contents
Page 2 - Deadlock and the four necessary conditions
Page 3 - Resource allocation graph and the Banker algorithm
Page 4 - Paging, page table and address translation
Page 5 - Page replacement algorithms and Belady anomaly
Page 6 - Thrashing and the working set model""",
    ),
    (
        "3.1 Deadlock and Its Necessary Conditions",
        """A deadlock is a situation in which a set of processes is blocked because each
process is holding a resource and waiting for another resource held by some
other process in the same set. No process in the set can ever proceed.

The four necessary conditions for deadlock, stated by Coffman, are:

1. Mutual exclusion: at least one resource is held in a non-sharable mode, so
   only one process can use it at a time.
2. Hold and wait: a process is holding at least one resource and is waiting to
   acquire additional resources that are currently held by other processes.
3. No preemption: a resource cannot be forcibly taken from a process; it is
   released only voluntarily by the process holding it.
4. Circular wait: a set of processes P0, P1, ..., Pn exists such that P0 waits
   for a resource held by P1, P1 waits for one held by P2, and Pn waits for one
   held by P0.

All four conditions must hold simultaneously for a deadlock to occur. Deadlock
prevention works by making sure at least one of the four conditions can never
hold. The four methods of handling deadlock are prevention, avoidance,
detection with recovery, and the ostrich approach of ignoring the problem.""",
    ),
    (
        "3.2 Resource Allocation Graph and Banker Algorithm",
        """A resource allocation graph is a directed graph with processes and resource
types as vertices. A request edge runs from a process to a resource type, and
an assignment edge runs from a resource instance to a process. If every
resource type has a single instance, a cycle in the graph means a deadlock. If
some resource type has multiple instances, a cycle is only a necessary
condition and the system may still be safe.

The Banker algorithm is a deadlock avoidance algorithm. Each process declares
in advance the maximum number of instances of each resource type it may need.
Before granting a request, the system checks whether granting it leaves the
system in a safe state, that is, whether some sequence of all processes exists
in which each process can obtain its maximum need, run, and release everything.

Data structures used: Available[m], Max[n][m], Allocation[n][m] and
Need[n][m] = Max - Allocation, for n processes and m resource types.

Safety algorithm
1. Work = Available and Finish[i] = false for all i.
2. Find an index i with Finish[i] = false and Need[i] <= Work.
3. Work = Work + Allocation[i], Finish[i] = true, and repeat step 2.
4. If Finish[i] is true for all i the state is safe, otherwise it is unsafe.

The request is granted only if the resulting state is safe; otherwise the
process must wait. The complexity of the safety check is O(m * n * n).""",
    ),
    (
        "3.3 Paging, Page Table and Address Translation",
        """Paging is a memory management scheme that removes the need for contiguous
allocation of physical memory and therefore removes external fragmentation.
Physical memory is divided into fixed size blocks called frames, and logical
memory is divided into blocks of the same size called pages.

Address translation
The CPU generates a logical address made of two parts: a page number p and a
page offset d. The page number indexes the page table, which stores the frame
number f for that page. The physical address is formed by combining the frame
number with the same offset d. For a page size of 2^n bytes, the low order n
bits of the logical address are the offset and the remaining high order bits
are the page number.

Every process has its own page table, pointed to by the page table base
register. A plain page table needs two memory accesses per data access: one for
the page table entry and one for the data itself. A translation lookaside
buffer, or TLB, is a small associative cache of recent page table entries that
removes the extra access on a hit. The effective access time depends on the TLB
hit ratio.

Page table entries also carry protection bits: a valid-invalid bit, read write
and execute permissions, a dirty bit and a reference bit. Internal
fragmentation of on average half a page per process remains, because the last
page of a process is usually only partly used.""",
    ),
    (
        "3.4 Page Replacement Algorithms",
        """When a page fault occurs and no free frame is available, the operating system
must select a victim page to evict. The aim of a page replacement algorithm is
to minimise the page fault rate for a given reference string.

FIFO page replacement
The oldest page in memory, the one that entered first, is replaced. FIFO is
simple to implement with a queue but performs poorly, because a page that has
been resident a long time may still be heavily used.

Optimal page replacement (OPT or MIN)
The page that will not be used for the longest period of time in the future is
replaced. It gives the lowest possible page fault rate, but it needs knowledge
of the future reference string, so it cannot be implemented. It is used as the
benchmark against which the other algorithms are measured.

LRU page replacement
The least recently used page is replaced, using the recent past as an
approximation of the near future. It is implemented with counters or with a
stack of page numbers, and it is close to optimal in practice but expensive in
hardware. Approximations used in practice include the second chance algorithm
and the clock algorithm, both based on the reference bit.

Belady anomaly
Belady anomaly is the counter-intuitive behaviour in which increasing the
number of allocated frames increases the number of page faults instead of
reducing it. FIFO suffers from Belady anomaly. Stack algorithms such as LRU and
OPT never suffer from it, because the set of pages in n frames is always a
subset of the set of pages in n+1 frames.""",
    ),
    (
        "3.5 Thrashing and the Working Set Model",
        """Thrashing is the condition in which a process spends more time paging than
executing. It happens when a process does not have enough frames to hold the
pages it is actively using, so every reference causes a page fault, which
evicts another page that is needed immediately afterwards.

Cause of thrashing
As the degree of multiprogramming rises, each process receives fewer frames.
Page fault rate climbs, CPU utilisation falls, and the scheduler responds by
admitting still more processes, which makes the situation worse. CPU
utilisation therefore collapses sharply beyond a critical degree of
multiprogramming.

Locality model
A locality is a set of pages that are used together. A process moves from one
locality to another as it executes. Thrashing is avoided if each process is
allocated enough frames to hold its current locality.

Working set model
The working set of a process is the set of pages referenced in the most recent
delta references, where delta is the working set window. The working set size
WSS(i) approximates the locality of process i. The system computes
D = sum of WSS(i) over all processes, the total demand for frames. If D is
greater than the number of available frames m, thrashing will occur, and the
operating system suspends one of the processes to relieve the pressure.

Page fault frequency is a more direct control: the system sets an upper and a
lower bound on the page fault rate of a process, gives it another frame when
the rate crosses the upper bound, and takes one away when it drops below the
lower bound.""",
    ),
]


def render_pdf(path: Path, pages: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for heading, body in pages:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_textbox(
            pymupdf.Rect(MARGIN, MARGIN, PAGE_W - MARGIN, MARGIN + 60),
            heading,
            fontsize=14,
            fontname="hebo",
        )
        rect = pymupdf.Rect(MARGIN, MARGIN + 50, PAGE_W - MARGIN, PAGE_H - MARGIN)
        for size in (11, 10, 9, 8):
            if page.insert_textbox(rect, body, fontsize=size, fontname="helv") >= 0:
                break
        else:  # pragma: no cover - would mean a page of text is far too long
            raise RuntimeError(f"text does not fit on a page: {heading}")
    doc.set_metadata(
        {
            "title": path.stem,
            "author": "VTU notes sample",
            "producer": "vtu-rag make_sample_pdfs",
            "creationDate": "D:20240101000000Z",
            "modDate": "D:20240101000000Z",
        }
    )
    doc.save(path, garbage=4, deflate=True)
    doc.close()


def main() -> int:
    pdf_dir = load_settings().pdf_dir
    targets = [
        (pdf_dir / "DBMS" / "DBMS_Module2.pdf", DBMS_PAGES),
        (pdf_dir / "OS" / "OS_Module3.pdf", OS_PAGES),
    ]
    for path, pages in targets:
        render_pdf(path, pages)
        print(f"[ok] {path}  ({len(pages)} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
