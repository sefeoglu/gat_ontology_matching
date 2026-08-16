from rdflib import BNode, Graph, Literal, Namespace, RDF, RDFS, URIRef


class OntologyParser(object):

    def __init__(self, rdf_file):
        self.graph = Graph()
        self.graph.parse(rdf_file, format=self._guess_format(rdf_file))
        self.classes = self.__get_classes()
        self.properties = self.__get_properties()

    @staticmethod
    def _guess_format(rdf_file):
        lower = rdf_file.lower()
        if lower.endswith('.ttl'):
            return 'turtle'
        if lower.endswith('.nt'):
            return 'nt'
        if lower.endswith('.jsonld'):
            return 'json-ld'
        if lower.endswith('.xml') or lower.endswith('.rdf'):
            return 'xml'
        return 'xml'

    def __normalize_uri(self, value):
        if value is None:
            return None
        value = str(value)
        return value.split('/')[-1].replace('*>', '').split('#')[-1]

    def __get_properties(self):
        properties = set()
        for _, predicate, _ in self.graph:
            if isinstance(predicate, (URIRef, BNode)):
                properties.add(self.__normalize_uri(predicate))
        return sorted(p for p in properties if p)

    def __get_classes(self):
        classes = set()
        for subject, predicate, obj in self.graph:
            if predicate == RDF.type and isinstance(obj, (URIRef, BNode)):
                if str(obj).endswith('Class') or str(obj).endswith('Thing') or obj == RDFS.Class:
                    classes.add(self.__normalize_uri(subject))
        return sorted(c for c in classes if c)

    def __get_all_triples(self):
        triples = []
        for subject, predicate, obj in self.graph:
            triples.append((str(subject), str(predicate), str(obj)))
        return triples

    def __get_inferred_properties(self, class_uri):
        class_value = class_uri
        if not isinstance(class_value, (URIRef, str)):
            return []
        if isinstance(class_value, str):
            try:
                class_value = URIRef(class_value)
            except Exception:
                return []

        return [
            str(prop)
            for _, prop, _ in self.graph.triples((class_value, None, None))
            if prop != RDF.type
        ]