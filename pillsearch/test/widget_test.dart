import 'package:flutter_test/flutter_test.dart';
import 'package:pillsearch/main.dart';

void main() {
  testWidgets('App smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const PerkIDApp());
    expect(find.text('PerkID — Pill Search'), findsOneWidget);
  });
}
