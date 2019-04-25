/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.gosololaw.elasticsearch;

import org.apache.lucene.index.*;

import org.apache.lucene.store.ByteArrayDataInput;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.plugins.ScriptPlugin;
import org.elasticsearch.script.ScoreScript;
import org.elasticsearch.script.ScoreScript.LeafFactory;
import org.elasticsearch.script.ScriptContext;
import org.elasticsearch.script.ScriptEngine;
import org.elasticsearch.search.lookup.SearchLookup;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Map;

/**
 * This class is instantiated when Elasticsearch loads the plugin for the
 * first time. If you change the name of this plugin, make sure to update
 * src/main/resources/es-plugin.properties file that points to this class.
 */
public final class VectorScoringPlugin extends Plugin implements ScriptPlugin {

    private static final int DOUBLE_SIZE = 8;

    @Override
    public ScriptEngine getScriptEngine(Settings settings, Collection<ScriptContext<?>> contexts) {
        return new VectorScoreEngine();
    }

    /**
     * An example {@link ScriptEngine} that uses Lucene segment details to implement pure document frequency scoring.
     */
    // tag::expert_engine
    private static class VectorScoreEngine implements ScriptEngine {

        @Override
        public String getType() {
            return "binary_vector_score";
        }

        @Override
        @SuppressWarnings("unchecked")
        public <T> T compile(String scriptName, String scriptSource, ScriptContext<T> context, Map<String, String> params) {
            if (!context.equals(ScoreScript.CONTEXT)) {
                throw new IllegalArgumentException(getType() + " scripts cannot be used for context [" + context.name + "]");
            }
            // we use the script "source" as the script identifier
            if ("vector_scoring".equals(scriptSource)) {
                ScoreScript.Factory factory = PureDfLeafFactory::new;
                return context.factoryClazz.cast(factory);
            }
            throw new IllegalArgumentException("Unknown script name " + scriptSource);
        }

        @Override
        public void close() {
            // optionally close resources
        }

        private static class PureDfLeafFactory implements LeafFactory {
            private final Map<String, Object> params;
            private final SearchLookup lookup;
            private final String field;
            private final double[] inputVector;
            private final double magnitude;

            @SuppressWarnings("unchecked")
            private PureDfLeafFactory(
                    Map<String, Object> params, SearchLookup lookup) {
                this.params = params;
                this.lookup = lookup;
                final Object field = params.get("vector_field");
                if (field == null)
                    throw new IllegalArgumentException("binary_vector_score script requires field vector_field");
                this.field = field.toString();

                // get query inputVector - convert to primitive
                final ArrayList<Double> tmp = (ArrayList<Double>) params.get("vector");
                this.inputVector = new double[tmp.size()];
                for (int i = 0; i < inputVector.length; i++) {
                    inputVector[i] = tmp.get(i);
                }

                // calc magnitude
                double queryVectorNorm = 0;
                // compute query inputVector norm once
                for (double v : inputVector) {
                    queryVectorNorm += v * v;
                }
                magnitude = Math.sqrt(queryVectorNorm);
            }

            @Override
            public boolean needs_score() {
                return false;  // Return true if the script needs the score
            }

            @Override
            public ScoreScript newInstance(LeafReaderContext context)
                    throws IOException {
                return new ScoreScript(params, lookup, context) {
                    BinaryDocValues accessor = context.reader().getBinaryDocValues(field);
                    Boolean is_value = false;

                    @Override
                    public void setDocument(int docId) {
                        try {
                            accessor.advanceExact(docId);
                            is_value = true;
                        } catch (IOException e) {
                            is_value = false;
                        }
                    }

                    @Override
                    public double execute() {
                        if (!is_value) return 0.0d;
                        final byte[] bytes;
                        try {
                            bytes = accessor.binaryValue().bytes;
                        } catch (IOException e) {
                            return 0.0d;
                        }

                        final int input_vector_size = inputVector.length;

                        final ByteArrayDataInput doc_vector = new ByteArrayDataInput(bytes);
                        doc_vector.readVInt(); // returns the number of values which should be 1, MUST appear hear since it affect the next calls
                        final int doc_vector_length = doc_vector.readVInt(); // returns the number of bytes to read
                        if (doc_vector_length != input_vector_size * DOUBLE_SIZE) {
                            return 0.0d;
                        }
                        final int position = doc_vector.getPosition();
                        final DoubleBuffer doubleBuffer = ByteBuffer.wrap(bytes, position, doc_vector_length).asDoubleBuffer();

                        final double[] docVector = new double[input_vector_size];
                        doubleBuffer.get(docVector);

                        double score = 0.0d;
                        double docVectorNorm = 0.0d;
                        for (int i = 0; i < input_vector_size; i++) {
                            docVectorNorm += docVector[i] * docVector[i];
                            score += docVector[i] * inputVector[i];
                        }
                        if (docVectorNorm == 0.0d || magnitude == 0.0d) {
                            return 0.0d;
                        } else {
                            return score / (Math.sqrt(docVectorNorm) * magnitude);
                        }
                    }
                };
            }
        }
    }
    // end::expert_engine
}
