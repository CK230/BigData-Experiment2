
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class InvertedIndex {

    // 第一部分：Mapper
    public static class IndexMapper
            extends Mapper<LongWritable, Text, Text, Text> {

        private final Set<String> stopwords = new HashSet<>();
        private final Text wordKey = new Text();
        private final Text fileValue = new Text();

        @Override
        protected void setup(Context context) throws IOException {

            Configuration conf = context.getConfiguration();

            Path stopwordPath =
                    new Path(conf.get("stopwords.path"));

            FileSystem fs = stopwordPath.getFileSystem(conf);

            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(
                            fs.open(stopwordPath),
                            StandardCharsets.UTF_8))) {

                String line;

                while ((line = reader.readLine()) != null) {
                    String word =
                            line.trim().toLowerCase(Locale.ROOT);

                    if (!word.isEmpty()) {
                        stopwords.add(word);
                    }
                }
            }

            // 获取当前文档的文件名
            FileSplit split =
                    (FileSplit) context.getInputSplit();

            fileValue.set(split.getPath().getName());
        }

        @Override
        protected void map(
                LongWritable key,
                Text value,
                Context context
        ) throws IOException, InterruptedException {

            String line =
                    value.toString().toLowerCase(Locale.ROOT);

            // 按非英文字母分词
            String[] words = line.split("[^a-z]+");

            for (String word : words) {

                if (word.isEmpty() || stopwords.contains(word)) {
                    continue;
                }

                wordKey.set(word);

                // 输出：单词 -> 文档名
                context.write(wordKey, fileValue);
            }
        }
    }

    // 第二部分：Reducer
    public static class IndexReducer
            extends Reducer<Text, Text, Text, Text> {

        @Override
        protected void reduce(
                Text key,
                Iterable<Text> values,
                Context context
        ) throws IOException, InterruptedException {

            Map<String, Integer> counts = new TreeMap<>();
            int total = 0;

            // 统计每篇文档中的出现次数
            for (Text value : values) {
                String filename = value.toString();

                counts.put(
                        filename,
                        counts.getOrDefault(filename, 0) + 1
                );

                total++;
            }

            StringBuilder result = new StringBuilder();

            // 拼接各文档词频
            for (Map.Entry<String, Integer> entry
                    : counts.entrySet()) {

                result.append("<")
                      .append(entry.getKey())
                      .append(",")
                      .append(entry.getValue())
                      .append(">;");
            }

            // 添加单词总词频
            result.append("<total,")
                  .append(total)
                  .append(">.");

            context.write(
                    key,
                    new Text(result.toString())
            );
        }
    }

    // 第三部分：程序入口
    public static void main(String[] args) throws Exception {

        if (args.length != 3) {
            System.err.println(
                "Usage: InvertedIndex <input> <output> <stopwords>"
            );
            System.exit(1);
        }

        Configuration conf = new Configuration();

        conf.set("stopwords.path", args[2]);

        Job job = Job.getInstance(
                conf,
                "Document Inverted Index"
        );

        job.setJarByClass(InvertedIndex.class);

        job.setMapperClass(IndexMapper.class);
        job.setReducerClass(IndexReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        job.setNumReduceTasks(1);

        FileInputFormat.addInputPath(
                job,
                new Path(args[0])
        );

        FileOutputFormat.setOutputPath(
                job,
                new Path(args[1])
        );

        System.exit(
                job.waitForCompletion(true) ? 0 : 1
        );
    }
}
